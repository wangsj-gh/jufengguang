.PHONY: all clean multi

LIB_DIR := -L/data/Database/cuda/lib64
LIB_DIR += -L/data/Database/gdal/lib

INCLUDE_DIR := -I/data/Database/cuda/include
INCLUDE_DIR += -I/data/Database/gdal/include

GCC := /data/Database/gcc/bin/g++
NVCC := /data/Database/cuda/bin/nvcc

LIBS := -lcudart
LIBS += -lpthread
LIBS += -lcuda
LIBS += -lgdal

CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:%.cu=%.o)

C_SRCS := $(wildcard *.cpp)
C_OBJS := $(C_SRCS:%.c=%.o)

EXECUTABLE := gdal.exe 

multi:
	$(MAKE) -j60 all

all : $(EXECUTABLE)

$(CU_OBJS):$(CU_SRCS)
	$(NVCC) -c $^

$(C_OBJS):$(C_SRCS)
	$(GCC) -std=c++11 -c $(LIBS) $^

$(EXECUTABLE):$(CU_OBJS) $(C_OBJS)
	$(GCC) -std=c++11 -o $@ $^ $(INCLUDE_DIR) $(LIB_DIR) $(LIBS)

clean:
	rm $(EXECUTABLE) *.o

