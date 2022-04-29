.PHONY: all clean multi

LIB_DIR := -L/usr/lib
LIB_DIR += -L/data/Database/cuda-11.4/lib64

INCLUDE_DIR := -I/usr/include
INCLUDE_DIR += -I/usr/gdal/include
INCLUDE_DIR += -I/data/Database/cuda-11.4/include

GCC := g++
NVCC := nvcc

LIBS := -lcudart
LIBS += -lcuda
LIBS += -lgdal
LIBS += -lnetcdf_c++4
LIBS += -lpthread

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

