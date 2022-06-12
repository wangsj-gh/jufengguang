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

__global__ void GetSunRiseSet(const double *Lat, const double *Doy, double *Sunrise, double *Sunset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double pi = 3.141592654;
    double delta = 23.45*sin((Doy[0]-80)*pi/180*360/365)*pi/180;
    double omegarise = -acos(-tan(Lat[i] * pi / 180) * tan(delta));
    double omegaset = acos(-tan(Lat[i] * pi / 180) * tan(delta));
    Sunrise[i] = 12 + omegarise * 180 / pi / 15;
    Sunset[i] = 12 + omegaset * 180 / pi / 15;
}

__global__ void GetLocalTime(const double *Lon, const double *timeSeries, double *LocalTime)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double time_zone = ceil((Lon[i] + 7.5) / 15.0) - 1;
    double local = fmod(timeSeries[0] + time_zone, 24.0);

    if (local < 0)
    {
        LocalTime[i] = local + 24;
    }
    else
    {
        LocalTime[i] = local;
    }
}

__global__ void GetCorrectionFactor(const double *Sunrise, const double *Sunset, const double *LocalTime, const int *DaySize, double *CorrectionFactor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double TempResolution = 24 / (double)DaySize[0];
    double halfTempResolution = TempResolution / 2.0;

    double upperSunrise = Sunrise[i] + halfTempResolution;
    double lowerSunrise = Sunrise[i] + halfTempResolution;

    double upperSunset = Sunset[i] + halfTempResolution;
    double lowerSunset = Sunset[i] + halfTempResolution;   

    if((LocalTime[i] < upperSunrise) || (lowerSunset < LocalTime[i]))
    {
        CorrectionFactor[i] = 0;
    }
    else if ((abs(LocalTime[i]-Sunrise[i]) < halfTempResolution))
    {
        CorrectionFactor[i]=abs(LocalTime[i]-Sunrise[i]) / TempResolution;
    }
    else if ((abs(LocalTime[i]-Sunset[i]) < halfTempResolution))
    {
        CorrectionFactor[i]=abs(LocalTime[i]-Sunset[i]) / TempResolution;
    }
    else if((abs(LocalTime[i]-Sunrise[i]) > halfTempResolution)  || (abs(Sunset[i] - LocalTime[i]) > halfTempResolution))
    {
        CorrectionFactor[i] = 1;
    }

}

__global__ void GetDayData(const double *DayGpp, const double *CorrectionFactor, const int *DaySize, double *GetDayGpp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double TempResolution = 24 / (double)DaySize[0];
    /// unit:umol co2/m2/day
    GetDayGpp[i] = DayGpp[i] * TempResolution * 3600.0 * CorrectionFactor[i];
    /// unit: kgC/m2/day
    GetDayGpp[i] = GetDayGpp[i] * 12.011 * pow(10, -9);
}

extern "C" void accum_C(int blocksPerGrid, int threadsPerBlock, 
                        const double *GPP, double *sum)
{
    accum<<<blocksPerGrid, threadsPerBlock>>>(GPP, sum);
}

extern "C" void GetMean_C(int blocksPerGrid, int threadsPerBlock, 
                            const double *SumGpp, const int *DaySize, double *MeanGpp)
{
    GetMean<<<blocksPerGrid, threadsPerBlock>>>(SumGpp, DaySize, MeanGpp);
}

extern "C" void GetSunRiseSet_C(int blocksPerGrid, int threadsPerBlock, 
                                const double *Lat, const double *Doy, 
                                double *Sunrise, double *Sunset)
{
    GetSunRiseSet<<<blocksPerGrid, threadsPerBlock>>>(Lat, Doy, Sunrise, Sunset);
}

extern "C" void GetLocalTime_C(int blocksPerGrid, int threadsPerBlock, 
                               const double *Lon, const double *timeSeries, double *LocalTime)
{
    GetLocalTime<<<blocksPerGrid, threadsPerBlock>>>(Lon, timeSeries, LocalTime);
}

extern "C" void GetCorrectionFactor_C(int blocksPerGrid, int threadsPerBlock, 
                                      const double *Sunrise, const double *Sunset, const double *LocalTime, const int *DaySize, 
                                      double *CorrectionFactor)
{
    GetCorrectionFactor<<<blocksPerGrid, threadsPerBlock>>>(Sunrise, Sunset, LocalTime, DaySize, CorrectionFactor);
}

extern "C" void GetDayData_C(int blocksPerGrid, int threadsPerBlock, const double *DayGpp, const double *CorrectionFactor, const int *DaySize,double *GetDayGpp)
{
    GetDayData<<<blocksPerGrid, threadsPerBlock>>>(DayGpp, CorrectionFactor,DaySize, GetDayGpp);
}