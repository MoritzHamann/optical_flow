COMPILER=clang++
FLAGS=-Wno-c++11-extensions -Wall -flto -O3
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc

hornschunck:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp misc.cpp tensor_computation.cpp hornschunck_with_gradient.cpp -o hornschunck

brox_iso:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp misc.cpp tensor_computation.cpp brox_iso.cpp -o brox_iso

brox_aniso:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp misc.cpp tensor_computation.cpp brox_aniso.cpp -o brox_aniso

hornschunck_separation:
	$(COMPILER) $(FLAGS) $(LIBS) main_separation.cpp misc.cpp initial_separation.cpp tensor_computation.cpp hornschunck_separation.cpp -o hornschunck_separation
