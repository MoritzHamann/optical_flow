COMPILER=clang++
FLAGS=-Wno-c++11-extensions -Wall -flto -O3
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc

hornschunck:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp tensor_computation.cpp hornschunck.cpp -o hornschunck

hornschunck_with_gradient:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp tensor_computation.cpp	hornschunck_with_gradient.cpp -o hornschunck_gradient

brox:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp tensor_computation.cpp brox.cpp -o brox
