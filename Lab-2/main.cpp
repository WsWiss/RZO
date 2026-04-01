#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const std::string imagePath = "image.png";
    const std::string outPath = "result.png";
    const cv::Scalar drawColor(0, 165, 255);
    const int thicknessFilled = -1;
    const int thicknessLine = 2;
    const int frameDelay = 80;

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "Изображение не найдено. Создаю пустой холст 800x600." << std::endl;
        image = cv::Mat(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    } else {
        std::cout << "Изображение загружено: " << imagePath << std::endl;
    }

    const std::string winName = "Результат";
    cv::namedWindow(winName);

    cv::line(image, cv::Point(50, 50), cv::Point(200, 50), drawColor, thicknessLine);
    cv::imshow(winName, image);
    cv::waitKey(frameDelay);

    cv::rectangle(image, cv::Point(250, 30), cv::Point(400, 120), drawColor, thicknessFilled);
    cv::imshow(winName, image);
    cv::waitKey(frameDelay);

    cv::circle(image, cv::Point(500, 75), 45, drawColor, thicknessFilled);
    cv::imshow(winName, image);
    cv::waitKey(frameDelay);

    cv::ellipse(image, cv::Point(150, 250), cv::Size(80, 50), 0, 0, 360, drawColor, thicknessFilled);
    cv::imshow(winName, image);
    cv::waitKey(frameDelay);

    std::vector<cv::Point> polygon = {
        cv::Point(350, 200),
        cv::Point(450, 200),
        cv::Point(500, 280),
        cv::Point(400, 320),
        cv::Point(300, 280)
    };
    std::vector<std::vector<cv::Point>> contours = { polygon };
    cv::fillPoly(image, contours, drawColor);
    cv::imshow(winName, image);
    cv::waitKey(frameDelay);

    const std::string text = "Bylov V.M. 22-PM-1";
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 1.2;
    const int textThickness = 2;
    int textX = 0;
    int textY = image.rows / 2;
    int dx = 4;
    int dy = 4;

    cv::Mat baseImage = image.clone();

    while (true) {
        cv::Mat frame = baseImage.clone();
        cv::putText(frame, text, cv::Point(textX, textY), fontFace, fontScale,
                    cv::Scalar(0, 0, 0), textThickness);

        cv::imshow(winName, frame);

        textX += dx;
        if (textX + 250 > frame.cols)
            dx = -dx;
        if (textX < 0)
            dx = -dx;
        
        textY += dy;
        if (textX + 250 > frame.cols)
            dy = -dy;
        if (textY < 0)
            dy = -dy;

        int key = cv::waitKey(30);
        if (key >= 0)
            break;
    }

    cv::Mat resultWithText = baseImage.clone();
    int bottomY = resultWithText.rows - 25;
    int leftX = 20;
    cv::putText(resultWithText, text, cv::Point(leftX, bottomY), fontFace, fontScale,
                cv::Scalar(0, 0, 0), textThickness);

    if (!cv::imwrite(outPath, resultWithText)) {
        std::cerr << outPath << std::endl;
        return -1;
    }
    std::cout << outPath << std::endl;

    cv::destroyAllWindows();
    return 0;
}
