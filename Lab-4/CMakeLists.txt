cmake_minimum_required(VERSION 3.10)
project(ComputerVision)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Поиск OpenCV
find_package(OpenCV REQUIRED)

# Вывод информации об OpenCV
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Добавление исполняемого файла
add_executable(app main.cpp)

# Подключение OpenCV
# Используем только необходимые библиотеки для базовых операций
target_link_libraries(app 
    opencv_core 
    opencv_imgcodecs 
    opencv_imgproc 
    opencv_highgui
)
target_include_directories(app PRIVATE ${OpenCV_INCLUDE_DIRS})
