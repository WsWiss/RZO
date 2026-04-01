#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sys/stat.h>

cv::Mat makeHueMask(const cv::Mat &hsv, const std::string &colorName) {
    std::string c = colorName;
    for (char &ch : c) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    cv::Mat mask;

    if (c == "red") {
        cv::Mat m1, m2;
        cv::inRange(hsv,
                    cv::Scalar(0,   70, 70),
                    cv::Scalar(15, 255, 255),
                    m1);
        cv::inRange(hsv,
                    cv::Scalar(160, 70, 70),
                    cv::Scalar(179, 255, 255),
                    m2);
        cv::bitwise_or(m1, m2, mask);
    } else if (c == "green") {
        cv::inRange(hsv,
                    cv::Scalar(35, 60, 60),
                    cv::Scalar(85, 255, 255),
                    mask);
    } else if (c == "blue") {
        cv::inRange(hsv,
                    cv::Scalar(90, 50, 40),
                    cv::Scalar(140, 255, 255),
                    mask);
    } else if (c == "yellow") {
        cv::inRange(hsv,
                    cv::Scalar(20, 70, 70),
                    cv::Scalar(40, 255, 255),
                    mask);
    } else {
        cv::inRange(hsv,
                    cv::Scalar(0, 70, 70),
                    cv::Scalar(15, 255, 255),
                    mask);
    }

    return mask; 
}

int main(int argc, char **argv) {
    cv::Mat img = cv::imread("../image.png");
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    mkdir("../out", 0755);

    const std::vector<std::string> colors = {"red", "green", "blue", "yellow"};
    int idx = 1;
    for (const auto &c : colors) {
        cv::Mat mask = makeHueMask(hsv, c);
        cv::imshow(c + " mask", mask);

        cv::Mat result;
        img.copyTo(result, mask);
        cv::imshow(c + " result", result);

        std::string prefix = "../out/" + std::to_string(idx);
        cv::imwrite(prefix + "_" + c + "_mask.png", mask);
        cv::imwrite(prefix + "_" + c + "_result.png", result);
        idx++;
    }

    while (true) {
        int key = cv::waitKey(0);
        if (key == 27) {
            break;
        }
    }

    return 0;
}