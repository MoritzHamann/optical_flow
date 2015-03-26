COMPILER=clang++
FLAGS=-Wno-c++11-extensions -Wall -flto -O3
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc

all:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp hornschunck.cpp tensor_computation.cpp -o main
