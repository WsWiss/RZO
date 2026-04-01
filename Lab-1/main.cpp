#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string imagePath = "image.png";
    const std::string outBase = "out/";

    auto saveImage = [](const std::string& path, const cv::Mat& img) {
        try {
            if (!cv::imwrite(path, img)) {
                std::cerr << "Ошибка: imwrite вернул false для: " << path << std::endl;
            } else {
                std::cout << "Сохранено: " << path << std::endl;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Ошибка: Не удалось сохранить " << path << "\n"
                      << e.what() << std::endl;
        }
    };
    
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Ошибка: Не удалось загрузить изображение по пути: " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Изображение успешно загружено: " << imagePath << std::endl;

    cv::imshow("Исходное изображение", image);
    saveImage(outBase + "01_original.png", image);

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::imshow("Градации серого", grayImage);
    saveImage(outBase + "02_gray.png", grayImage);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::imshow("Изменение размера", resizedImage);
    saveImage(outBase + "03_resized.png", resizedImage);

    cv::Mat rotatedImage;
    cv::rotate(image, rotatedImage, cv::ROTATE_90_CLOCKWISE);
    cv::imshow("Поворот", rotatedImage);
    saveImage(outBase + "04_rotated.png", rotatedImage);

    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(15, 15), 0);
    cv::imshow("Размытие", blurredImage);
    saveImage(outBase + "05_blurred.png", blurredImage);

    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);
    cv::imshow("Границы", edges);
    saveImage(outBase + "06_edges.png", edges);


    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}