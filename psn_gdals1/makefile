.PHONY: all clean multi

LIB_DIR := -L/data/Database/cuda/lib64
LIB_DIR += -L/data/Database/gdal/lib
LIB_DIR += -L/data/Database/netcdf-c/4.7.4-hdf5-1.12.1/lib


INCLUDE_DIR := -I/data/Database/cuda/include
INCLUDE_DIR += -I/data/Database/gdal/include
INCLUDE_DIR += -I/data/Database/netcdf-c/4.7.4-hdf5-1.12.1/include
INCLUDE_DIR += -I./netcdf/cxx4

GCC := /data/Database/gcc/bin/g++
NVCC := /data/Database/cuda/bin/nvcc

LIBS := -lcudart
LIBS += -lcuda
LIBS += -lgdal
LIBS += -lnetcdf
LIBS += -lpthread

CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:%.cu=%.o)

netcdf_SRCS := $(wildcard ./netcdf/cxx4/*.cpp)
netcdf_OBJS := $(netcdf_SRCS:%.c=%.o)

C_SRCS := $(wildcard *.cpp)
C_OBJS := $(C_SRCS:%.c=%.o)

EXECUTABLE := gdal.exe 

multi:
	$(MAKE) -j60 all

all : $(EXECUTABLE)

$(CU_OBJS):$(CU_SRCS)
	$(NVCC) -arch=compute_80 -code=sm_80 -c $^

$(C_OBJS):$(C_SRCS)
	$(GCC) -std=c++11 -c $(LIBS) $^

$(netcdf_OBJS):$(netcdf_SRCS)
	$(GCC) -std=c++11 -c $(LIBS) $^

$(EXECUTABLE):$(CU_OBJS) $(C_OBJS) $(netcdf_OBJS)
	$(GCC) -std=c++11 -o $@ $^ $(INCLUDE_DIR) $(LIB_DIR) $(LIBS)

clean:
	rm $(EXECUTABLE) *.o

