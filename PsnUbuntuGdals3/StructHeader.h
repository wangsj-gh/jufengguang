#pragma once
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <deque>

#include "omp.h"
#include <thread>
using namespace std;

struct VarInfo
{
    VarInfo()
    {
        ID = -1;
        Name = "none";
        LongName = "none";
        DataType = "none";

        scale_factor = 0.0f;
        add_offset = 0.0f;
        fill_value = 0.0f;
        missing_value = 0.0f;

        DimsCount = 0;
        DimsName = NULL;
        DimsSize = NULL;

        IsReadData = false;
        // Data = NULL;
    }
    int ID;
    std::string Name;
    std::string LongName;
    std::string DataType;

    float scale_factor;
    float add_offset;
    float fill_value;
    float missing_value;

    int DimsCount;
    std::string *DimsName;
    int *DimsSize;

    bool IsReadData;
    std::deque<unsigned char *> Data;

    // unsigned char* Data;
};

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
    double *GeoTransform;
    const char *Projection;
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
    double *p0;
    double *p1;
    double *p2;
    double *p3;
    double *p4;
    double *p5;

    double *Lon;
    double *Lat;
};

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
    double *timeSeries;
};

struct InformStruct
{
    InformStruct()
    {
        TifRaster = NULL;
        LaiBand = NULL;
    }
    TifRasterStruct *TifRaster;
    LaiBandStruct *LaiBand;
};

//����ģ�͵�����  index,Lai,Pressure,Rad,SH,Temp
// struct DataSetRead
struct DatasetInputStruct
{
    DatasetInputStruct()
    {
        Pressure = NULL;
        Rad = NULL;
        SH = NULL;
        Temp = NULL;
    }
    double *Pressure;
    double *Rad;
    double *SH;
    double *Temp;
};
