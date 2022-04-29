#pragma once
#include <gdal.h>
#include <gdal_priv.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <deque>

#include "omp.h"
#include <thread>
using namespace std;

//Raster数据结构体
struct TifRasterStruct
{
    TifRasterStruct()
    {
        RasterXSize = -1;
        RasterYSize = -1;
        RasterCount = -1;
        GeoTransform = NULL;
        Projection = NULL;
    }
    int RasterXSize;
    int RasterYSize;
    int RasterCount;
    double* GeoTransform;
    const char* Projection;
};

struct LaiBandStruct
{
    LaiBandStruct()
    {
        p0 = NULL;
        p1 = NULL;
        p2 = NULL;
        p3 = NULL;
        p4 = NULL;
        p5 = NULL;

        Lon = NULL;
        Lat = NULL;
    }
    double* p0;
    double* p1;
    double* p2;
    double* p3;
    double* p4;
    double* p5;

    double* Lon;
    double* Lat;
};

//文件的,时间分辨率,空间分辨率
struct SpatiotemporalStruct
{
    SpatiotemporalStruct()
    {
        TempResolution = 0;
        SpaResolution = 0;
        TimeSize = 0;
        timeSeries = NULL;
    }

    int TempResolution;
    int SpaResolution;
    int TimeSize;
    double* timeSeries;
};

struct InformStruct
{
    InformStruct()
    {
        TifRaster = NULL;
        LaiBand = NULL;
        Spatiotemporal = NULL;
    }
    TifRasterStruct* TifRaster;
    LaiBandStruct* LaiBand;
    SpatiotemporalStruct* Spatiotemporal;
};

//输入模型的数据  index,Lai,Pressure,Rad,SH,Temp
//struct DataSetRead
struct DatasetInputStruct
{
    DatasetInputStruct()
    {
        Pressure = NULL;
        Rad = NULL;
        SH = NULL;
        Temp = NULL;
    }
    double* Pressure;
    double* Rad;
    double* SH;
    double* Temp;
};





