CXX = g++
CXXFLAGS = -std=c++11 -Wall
TARGET = app
SOURCE = main.cpp

# Получение флагов компиляции через pkg-config
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)

# Используем только необходимые библиотеки для базовых операций с изображениями
# Это избегает проблем с зависимостями VTK и HDF5
OPENCV_LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(SOURCE) -o $(TARGET) $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
