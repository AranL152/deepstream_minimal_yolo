TRT  ?= /home/aranl/TensorRT-8.6.1.6
CUDA ?= /usr/local/cuda-12.6

CXXFLAGS = -O2 -std=c++17 -Wall -Wno-deprecated-declarations -Icar \
           -I$(TRT)/include -I$(CUDA)/include `pkg-config --cflags opencv4`
LDLIBS   = -L$(TRT)/lib -lnvinfer -L$(CUDA)/lib64 -lcudart `pkg-config --libs opencv4` \
           -Wl,-rpath,$(TRT)/lib

bench_car: bench_car.cpp car/deep_inference.cpp car/deep_inference.hpp
	g++ $(CXXFLAGS) -o $@ bench_car.cpp car/deep_inference.cpp $(LDLIBS)

clean:
	rm -f bench_car
