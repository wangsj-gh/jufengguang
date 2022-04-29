LIB_DIR = -L/data/Database/netcdf-c/4.7.4-hdf5-1.12.1/lib

INCLUDE_DIR = -I/data/Database/netcdf-c/4.7.4-hdf5-1.12.1/include

GCC := /data/Database/gcc/bin/g++

LIBS = -lnetcdf
LIBS += -lpthread

netcdf_SRCS := $(wildcard *.cpp)
netcdf_OBJS := $(netcdf_SRCS:%.c=%.o)

$(netcdf_OBJS):$(netcdf_SRCS)
	$(GCC) -std=c++11 -c $(LIBS) $^ -o ../../$@