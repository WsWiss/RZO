#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>

std::string classifyShape(const std::vector<cv::Point> &contour,
                          const std::vector<cv::Point> &approx) {
    int vertices = static_cast<int>(approx.size());
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);

    if (vertices == 3) return "Triangle";

    if (vertices == 4) {
        cv::Rect br = cv::boundingRect(approx);
        double aspect = static_cast<double>(br.width) / br.height;

        double minCos = 1.0, maxCos = -1.0;
        for (int i = 0; i < 4; i++) {
            cv::Point v1 = approx[(i + 1) % 4] - approx[i];
            cv::Point v2 = approx[(i + 3) % 4] - approx[i];
            double cosAngle = (v1.x * v2.x + v1.y * v2.y) /
                              (cv::norm(v1) * cv::norm(v2));
            minCos = std::min(minCos, cosAngle);
            maxCos = std::max(maxCos, cosAngle);
        }

        bool rightAngles = std::abs(minCos) < 0.15 && std::abs(maxCos) < 0.15;

        if (rightAngles) {
            if (aspect > 0.85 && aspect < 1.15)
                return "Square";
            return "Rectangle";
        }
        return "Rhombus";
    }

    if (vertices == 5) return "Pentagon";
    if (vertices == 6) return "Hexagon";

    if (vertices > 6 && circularity > 0.75) {
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        double axisRatio = std::min(ellipse.size.width, ellipse.size.height) /
                           std::max(ellipse.size.width, ellipse.size.height);
        if (axisRatio > 0.85)
            return "Circle";
        return "Ellipse";
    }

    return "Polygon(" + std::to_string(vertices) + ")";
}

int main(int argc, char **argv) {
    std::string imagePath = "image.png";
    if (argc >= 2) {
        imagePath = argv[1];
    }

    cv::Mat img = cv::imread(imagePath);
    if (img.empty() && argc < 2) {
        imagePath = "../image.png";
        img = cv::imread(imagePath);
    }
    if (img.empty()) {
        std::cerr << "Cannot load image: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);

    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat result = img.clone();
    int shapeNum = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 500) continue;

        double perimeter = cv::arcLength(contours[i], true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.02 * perimeter, true);

        std::string shapeName = classifyShape(contours[i], approx);

        cv::Moments M = cv::moments(contours[i]);
        int cx = static_cast<int>(M.m10 / M.m00);
        int cy = static_cast<int>(M.m01 / M.m00);

        cv::drawContours(result, contours, static_cast<int>(i),
                         cv::Scalar(0, 255, 0), 2);
        cv::circle(result, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(result, shapeName, cv::Point(cx - 40, cy - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        cv::putText(result, "(" + std::to_string(cx) + "," + std::to_string(cy) + ")",
                    cv::Point(cx - 40, cy + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);

        shapeNum++;
        std::cout << shapeNum << ". " << shapeName
                  << "  center=(" << cx << ", " << cy << ")"
                  << "  area=" << area
                  << "  vertices=" << approx.size() << std::endl;
    }

    if (mkdir("../out", 0755) != 0 && errno != EEXIST) {
        std::perror("Cannot create out directory");
    }
    cv::imwrite("../out/01_original.png", img);
    cv::imwrite("../out/02_edges_canny.png", edges);
    cv::imwrite("../out/03_result.png", result);

    cv::imshow("Original", img);
    cv::imshow("Edges (Canny)", edges);
    cv::imshow("Result", result);

    while (true) {
        if (cv::waitKey(0) == 27) break;
    }

    return 0;
}