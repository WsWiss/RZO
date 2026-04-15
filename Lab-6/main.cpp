// Lab-6: Haar cascades — face on full frame; eyes and smile only inside face ROI.

#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <clocale>
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace cv;
using namespace std;

#ifdef _WIN32
static string moduleDir() {
  char p[MAX_PATH];
  if (!GetModuleFileNameA(nullptr, p, MAX_PATH))
    return ".";
  string s(p);
  size_t k = s.find_last_of("\\/");
  return (k == string::npos) ? string(".") : s.substr(0, k);
}
#else
static string moduleDir() { return "."; }
#endif

static bool openVideoFile(VideoCapture &cap, const string &path) {
  cap.release();
  if (cap.open(path, CAP_FFMPEG))
    return true;
#ifdef _WIN32
  cap.release();
  if (cap.open(path, CAP_MSMF))
    return true;
#endif
  cap.release();
  if (cap.open(path, CAP_ANY))
    return true;
  cap.release();
  return cap.open(path);
}

static bool loadCascade(CascadeClassifier &cc, const string &name) {
  const vector<string> tries = {name, moduleDir() + "/" + name, "../" + name};
  for (const string &t : tries) {
    if (cc.load(t))
      return true;
  }
  return false;
}

int main(int argc, char **argv) {
  setlocale(LC_ALL, "ru");

  string videoPath = "ZUA.mp4";
  string outPath = "result_lab7.mp4";
  if (argc > 1)
    videoPath = argv[1];
  if (argc > 2)
    outPath = argv[2];

  CascadeClassifier faceCascade, eyeCascade, smileCascade;
  VideoCapture cap;

  if (!loadCascade(faceCascade, "haarcascade_frontalface_default.xml")) {
    cerr << "Failed to load haarcascade_frontalface_default.xml\n";
    return -1;
  }
  if (!loadCascade(eyeCascade, "haarcascade_eye.xml")) {
    cerr << "Failed to load haarcascade_eye.xml\n";
    return -1;
  }
  if (!loadCascade(smileCascade, "haarcascade_smile.xml")) {
    cerr << "Failed to load haarcascade_smile.xml\n";
    return -1;
  }

  if (!openVideoFile(cap, videoPath) &&
      !openVideoFile(cap, moduleDir() + "/" + videoPath)) {
    cerr << "Could not open video: " << videoPath << endl;
    return -1;
  }

  const int w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
  const int h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
  double fpsIn = cap.get(CAP_PROP_FPS);
  if (!(fpsIn > 0.0 && fpsIn < 240.0))
    fpsIn = 30.0;

  VideoWriter writer;
  if (!writer.open(outPath, VideoWriter::fourcc('m', 'p', '4', 'v'), fpsIn, Size(w, h),
                   true)) {
    if (!writer.open(outPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsIn, Size(w, h),
                     true)) {
      cerr << "Could not open VideoWriter: " << outPath << endl;
      return -1;
    }
  }

  setUseOptimized(true);
  setNumThreads(getNumberOfCPUs());

  deque<Mat> frameQueue;
  mutex queueMutex;
  condition_variable queueCv;
  const size_t maxQueue = 3;
  bool stop = false;
  bool readDone = false;

  thread reader([&] {
    Mat f;
    while (true) {
      {
        lock_guard<mutex> lk(queueMutex);
        if (stop)
          break;
      }

      if (!cap.read(f))
        break;

      unique_lock<mutex> lk(queueMutex);
      queueCv.wait(lk, [&] { return stop || frameQueue.size() < maxQueue; });
      if (stop)
        break;
      frameQueue.push_back(f.clone());
      lk.unlock();
      queueCv.notify_one();
    }

    lock_guard<mutex> lk(queueMutex);
    readDone = true;
    queueCv.notify_all();
  });

  Mat frame, gray, graySmall, graySmallPad;
  vector<Rect> facesSmall;
  const double detectScale = 0.60;
  const int detectEvery = 1;      
  const int maxFacesToProcess = 3;
  const int detectPad = 24;  
  int frameIdx = 0;

  vector<Rect> cachedFaces;
  vector<vector<Rect>> cachedEyesAbs;
  vector<Rect> cachedSmileAbs;
  vector<bool> cachedHasSmile;
  vector<int> cachedSmileMisses;
  const int keepSmileFrames = 4;

  double fpsEma = 0.0;
  const double fpsAlpha = 0.10;

  while (true) {
    {
      unique_lock<mutex> lk(queueMutex);
      queueCv.wait(lk, [&] { return stop || !frameQueue.empty() || readDone; });
      if (stop || (frameQueue.empty() && readDone))
        break;
      frame = frameQueue.front();
      frameQueue.pop_front();
    }
    queueCv.notify_one();

    const int64 t0 = getTickCount();

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    if (frameIdx % detectEvery == 0 || cachedFaces.empty()) {
      resize(gray, graySmall, Size(), detectScale, detectScale, INTER_LINEAR);
      copyMakeBorder(graySmall, graySmallPad, detectPad, detectPad, detectPad, detectPad,
                     BORDER_REPLICATE);
      faceCascade.detectMultiScale(graySmallPad, facesSmall, 1.12, 8, 0, Size(30, 30));

      vector<Rect> allFaces;
      allFaces.reserve(facesSmall.size());
      for (const Rect &fs : facesSmall) {
        Rect face(cvRound((fs.x - detectPad) / detectScale),
                  cvRound((fs.y - detectPad) / detectScale),
                  cvRound(fs.width / detectScale), cvRound(fs.height / detectScale));
        face &= Rect(0, 0, gray.cols, gray.rows);
        if (face.width >= 40 && face.height >= 40)
          allFaces.push_back(face);
      }

      cachedFaces.clear();
      cachedEyesAbs.clear();
      cachedSmileAbs.clear();
      cachedHasSmile.clear();

      for (int k = 0; k < maxFacesToProcess && !allFaces.empty(); ++k) {
        int best = -1, bestArea = -1;
        for (int i = 0; i < static_cast<int>(allFaces.size()); ++i) {
          const int a = allFaces[i].area();
          if (a > bestArea) {
            bestArea = a;
            best = i;
          }
        }
        if (best < 0)
          break;
        cachedFaces.push_back(allFaces[best]);
        allFaces.erase(allFaces.begin() + best);
      }

      cachedEyesAbs.resize(cachedFaces.size());
      cachedSmileAbs.resize(cachedFaces.size());
      if (cachedHasSmile.size() != cachedFaces.size()) {
        cachedHasSmile.assign(cachedFaces.size(), false);
        cachedSmileMisses.assign(cachedFaces.size(), keepSmileFrames);
      }

      for (size_t fi = 0; fi < cachedFaces.size(); ++fi) {
        const Rect &face = cachedFaces[fi];

        Rect eyeRoi(face.x, face.y, face.width, cvRound(face.height * 0.55));
        eyeRoi &= Rect(0, 0, gray.cols, gray.rows);
        vector<Rect> eyes;
        eyeCascade.detectMultiScale(gray(eyeRoi), eyes, 1.12, 12, 0,
                                    Size(max(16, face.width / 8), max(16, face.height / 8)));

        int best1 = -1, best2 = -1, a1 = -1, a2 = -1;
        for (int i = 0; i < static_cast<int>(eyes.size()); ++i) {
          const int a = eyes[i].area();
          if (a > a1) {
            a2 = a1;
            best2 = best1;
            a1 = a;
            best1 = i;
          } else if (a > a2) {
            a2 = a;
            best2 = i;
          }
        }
        for (int idx : {best1, best2}) {
          if (idx >= 0) {
            Rect e = eyes[idx];
            cachedEyesAbs[fi].push_back(
                Rect(eyeRoi.x + e.x, eyeRoi.y + e.y, e.width, e.height));
          }
        }

        Rect mouthRoi(face.x + cvRound(face.width * 0.10),
                      face.y + cvRound(face.height * 0.50), cvRound(face.width * 0.80),
                      cvRound(face.height * 0.45));
        mouthRoi &= Rect(0, 0, gray.cols, gray.rows);
        if (mouthRoi.width < 20 || mouthRoi.height < 12)
          continue;

        vector<Rect> smiles;
        smileCascade.detectMultiScale(
            gray(mouthRoi), smiles, 1.22, 18, 0,
            Size(max(20, mouthRoi.width / 4), max(12, mouthRoi.height / 5)));

        Rect bestSmile;
        int bestSmileArea = -1;
        for (const Rect &s : smiles) {
          if (s.width < s.height)
            continue;
          const int area = s.area();
          if (area > bestSmileArea) {
            bestSmileArea = area;
            bestSmile = s;
          }
        }
        if (bestSmileArea > 0) {
          cachedSmileAbs[fi] = Rect(mouthRoi.x + bestSmile.x, mouthRoi.y + bestSmile.y,
                                    bestSmile.width, bestSmile.height);
          cachedHasSmile[fi] = true;
          cachedSmileMisses[fi] = 0;
        } else if (cachedHasSmile[fi] && cachedSmileMisses[fi] < keepSmileFrames) {
          ++cachedSmileMisses[fi];
        } else {
          cachedHasSmile[fi] = false;
        }
      }
    }

    for (size_t fi = 0; fi < cachedFaces.size(); ++fi) {
      rectangle(frame, cachedFaces[fi], Scalar(0, 255, 0), 2);
      for (const Rect &eye : cachedEyesAbs[fi]) {
        Point c(eye.x + eye.width / 2, eye.y + eye.height / 2);
        int r = max(3, min(eye.width, eye.height) / 2);
        circle(frame, c, r, Scalar(255, 0, 0), 2);
      }
      if (cachedHasSmile[fi])
        rectangle(frame, cachedSmileAbs[fi], Scalar(0, 0, 255), 2);
    }

    const double dt = (getTickCount() - t0) / getTickFrequency();
    const double fps = (dt > 0.0) ? (1.0 / dt) : 0.0;
    fpsEma = (fpsEma <= 0.0) ? fps : (fpsAlpha * fps + (1.0 - fpsAlpha) * fpsEma);
    putText(frame, format("FPS: %.1f", fpsEma), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9,
            Scalar(0, 255, 255), 2, LINE_AA);
    cout << "\rFPS: " << fixed << setprecision(1) << fpsEma << "   " << flush;

    writer.write(frame);
    imshow("Lab-7: pipeline read->process", frame);
    if (waitKey(1) == 27) {
      lock_guard<mutex> lk(queueMutex);
      stop = true;
      queueCv.notify_all();
      break;
    }
    ++frameIdx;
  }

  {
    lock_guard<mutex> lk(queueMutex);
    stop = true;
    queueCv.notify_all();
  }
  if (reader.joinable())
    reader.join();

  writer.release();
  cap.release();
  destroyAllWindows();
  return 0;
}
