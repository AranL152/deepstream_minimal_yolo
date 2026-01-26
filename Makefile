CC = g++
CFLAGS = -Wall -std=c++11
PKGS = gstreamer-1.0
INCS = -I/opt/nvidia/deepstream/deepstream/sources/includes \
       -I/usr/local/cuda-12.6/include
LIBS = -L/opt/nvidia/deepstream/deepstream/lib \
       -lnvdsgst_meta -lnvds_meta \
       -L/usr/local/cuda-12.6/lib64 -lcudart \
       -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib

all:
	$(CC) $(CFLAGS) -o minimal_yolo minimal_yolo.cpp \
	`pkg-config --cflags $(PKGS)` $(INCS) \
	`pkg-config --libs $(PKGS)` $(LIBS)

clean:
	rm -f minimal_yolo
