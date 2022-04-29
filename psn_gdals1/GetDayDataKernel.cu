/*The getDayData model*/
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "StructHeader.h"

using namespace std;

__global__ void accum(const double *GPP, double *sum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    sum[i] = sum[i] + GPP[i];
}

__global__ void GetMean(const double *SumGpp, const int *DaySize, double *MeanGpp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    MeanGpp[i] = SumGpp[i] / DaySize[0];
}

/*daytime*/
__global__ void DayLength(const double *Lat, const double *daydoy, double *daylength)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double pi = 3.141592654;
    double trans = 0.01745329;
    double delta = 23.45 * trans * sin((daydoy[0] - 80) / 365 * 360 * trans);
    double costou = -tan(Lat[i] * pi / 180) * tan(delta);
    double temp = -costou / sqrt(1 - pow(costou, 2));
    double tau = atanf(temp) + 2 * atanf(1);
    daylength[i] = tau / trans * 2 * 3600 / 15;
    // daylength[i] = 24.0 * 3600.0;
}

__global__ void GetDayData(const double *DayGpp, const double *Lai, const double *DayLength, double *GetDayGpp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    /// unit:umol co2/m2/day
    // GetDayGpp[i] = DayGpp[i] * Lai[i] * DayLength[i];
    // GetDayGpp[i] = DayGpp[i] * Lai[i] * 24.0 * 3600.0;
    GetDayGpp[i] = DayGpp[i] * 24.0 * 3600.0;
    /// unit: kgC/m2/day
    GetDayGpp[i] = GetDayGpp[i] * 12.011 * pow(10, -9);
}

extern "C" void accum_C(int blocksPerGrid, int threadsPerBlock, const double *GPP, double *sum)
{
    accum<<<blocksPerGrid, threadsPerBlock>>>(GPP, sum);
}

extern "C" void GetMean_C(int blocksPerGrid, int threadsPerBlock, const double *SumGpp, const int *DaySize, double *MeanGpp)
{
    GetMean<<<blocksPerGrid, threadsPerBlock>>>(SumGpp, DaySize, MeanGpp);
}

extern "C" void DayLength_C(int blocksPerGrid, int threadsPerBlock, const double *Lat, const double *daydoy, double *daylength)
{
    DayLength<<<blocksPerGrid, threadsPerBlock>>>(Lat, daydoy, daylength);
}

extern "C" void GetDayData_C(int blocksPerGrid, int threadsPerBlock, const double *DayGpp, const double *Lai, const double *DayLength, double *GetDayGpp)
{
    GetDayData<<<blocksPerGrid, threadsPerBlock>>>(DayGpp, Lai, DayLength, GetDayGpp);
}