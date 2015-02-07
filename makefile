COMPILER=clang++
FLAGS=-Wno-c++11-extensions -Wall -O3
FILES=flow_utility.cpp lodepng.cpp filehandling.cpp image_class.cpp hornschunck.cpp main.cpp


all:
	$(COMPILER) $(FLAGS) $(FILES) -o main

run: all
	./main

clean:
	rm main
