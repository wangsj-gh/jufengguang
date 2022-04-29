#pragma once
#include "StructHeader.h"
#include "PSN.h"

extern "C" void accum_C(int blocksPerGrid, int threadsPerBlock, const double *GPP,
                        double *sum);
extern "C" void GetMean_C(int blocksPerGrid, int threadsPerBlock, const double *SumGpp, const int *DaySize,
                          double *MeanGpp);
extern "C" void DayLength_C(int blocksPerGrid, int threadsPerBlock, const double *Lat, const double *daydoy,
                            double *daylength);

extern "C" void GetDayData_C(int blocksPerGrid, int threadsPerBlock, const double *DayGpp, const double *Lai, const double *DayLength,
                             double *GetDayGpp);

class GetDayData
{
public:
    void Init(std::deque<InformStruct *> *OutLaiInputDeque,
              std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque);
    void Inneed();
    void GetDayDataGPU(std::map<std::string, std::deque<VarInfo *> *> TleafDeque,
                       std::deque<double *> *Output);
    void GetDayDataGPU(std::deque<double *> *DataDeque,
                       std::deque<double *> *Output);
    void GetYearDataGPU(std::deque<double *> *DataDeque,
                        std::deque<double *> *Output);
    void GetDayDataGPU(std::deque<double *> *GppDeque,
                       std::deque<InformStruct *> *OutLaiInputDeque,
                       std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque,
                       std::deque<double *> *Output);
    void Release();

private:
    std::deque<InformStruct *> *OutLaiInputDeque;
    std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque;
    std::map<std::string, std::deque<VarInfo *> *> TleafDeque;
    std::deque<double *> *DataDeque;
    std::deque<double *> *GppDeque;

    int h_DaySize;

    ///////////////device////////////////
    double *d_sum;
    double *d_Gpp;
    int *d_DaySize;
    double *d_MeanGpp;

    double *d_DayGPP;

    double *d_p0;
    double *d_p1;
    double *d_p2;
    double *d_p3;
    double *d_p4;
    double *d_p5;
    // double *d_tiemsers;
    double *d_doy;

    double *d_Lai;

    double *d_Lat;

    double *d_daylength;

    cudaError_t err;
    size_t size;
    int numElements;
};