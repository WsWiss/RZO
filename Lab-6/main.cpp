
#include <opencv2/opencv.hpp>

#include <clocale>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

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

int main(int argc, char **argv) {
  setlocale(LC_ALL, "ru");

  string path = "Motion abstract geometric shapes.mkv";
  string outPath = "result.mp4";
  if (argc > 1)
    path = argv[1];
  if (argc > 2)
    outPath = argv[2];

  VideoCapture cap;
  if (!openVideoFile(cap, path)) {
    cerr << "Could not open video: " << path << endl;
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

  Mat frame, gray, binary;
  vector<vector<Point>> contours;

  while (cap.read(frame)) {
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 230, 255, THRESH_BINARY_INV);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);
    findContours(binary, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Mat result = frame.clone();

    for (const auto &contour : contours) {
      double perimeter = arcLength(contour, true);
      double area = contourArea(contour);
      if (area < 300)
        continue;

      vector<Point> approx;
      double epsilon = 0.07 * perimeter;
      approxPolyDP(contour, approx, epsilon, true);

      string shapeName;
      Scalar color;

      if (approx.size() == 3) {
        shapeName = "Triangle";
        color = Scalar(255, 0, 0);
      } else if (approx.size() == 4) {
        Rect rect = boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;
        if (aspectRatio >= 0.9 && aspectRatio <= 1.1) {
          shapeName = "Square";
          color = Scalar(0, 255, 0);
        } else {
          shapeName = "Rectangle";
          color = Scalar(255, 255, 0);
        }
      } else {
        if (perimeter < 1e-6)
          continue;
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity > 0.75) {
          shapeName = "Circle";
          color = Scalar(0, 0, 255);
        } else {
          continue;
        }
      }

      Rect bbox = boundingRect(contour);
      Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
      rectangle(result, bbox, color, 2);
      circle(result, center, 5, color, -1);
      int baseline = 0;
      Size textSize =
          getTextSize(shapeName, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
      Point textPos(center.x - textSize.width / 2, center.y - 10);
      rectangle(result,
                Rect(textPos.x - 5, textPos.y - textSize.height - 5,
                     textSize.width + 10, textSize.height + 10),
                color, -1);
      putText(result, shapeName, textPos, FONT_HERSHEY_SIMPLEX, 0.6,
              Scalar(255, 255, 255), 2);
      cout << "Detected: " << shapeName << " | vertices: " << approx.size()
           << " | area: " << area << " | center: (" << center.x << ", "
           << center.y << ")" << endl;
    }

    writer.write(result);
    imshow("Shape detection", result);
    imshow("Binary mask", binary);
    if (waitKey(30) == 27)
      break;
  }

  writer.release();
  cap.release();
  destroyAllWindows();
  return 0;
}
