#include "PSN.h"
#include "GetDayData.h"
#include <iostream>

using namespace std;

void GetDayData::Init(std::deque<InformStruct *> *OutLaiInputDeque,
                      std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque)
{
    TifRasterStruct *p_raster = OutLaiInputDeque->back()->TifRaster;
    numElements = p_raster->RasterXSize * p_raster->RasterYSize;
    size = numElements * sizeof(double);

    h_DaySize = SpatiotemporalDeque->back()->TimeSize;

    double *d_sum = NULL;
    double *d_Gpp = NULL;
    double *d_DaySize = NULL;
    double *d_MeanGpp = NULL;

    double *d_DayGPP = NULL;

    double *d_p0 = NULL;
    double *d_p1 = NULL;
    double *d_p2 = NULL;
    double *d_p3 = NULL;
    double *d_p4 = NULL;
    double *d_p5 = NULL;
    // double *d_tiemsers = NULL;
    double *d_doy = NULL;

    double *d_Lai = NULL;

    double *d_Lat = NULL;
}

void GetDayData::Inneed()
{
    err = cudaMalloc((void **)&d_sum, size);
    err = cudaMalloc((void **)&d_Gpp, size);
    err = cudaMalloc((void **)&d_DaySize, size);
    err = cudaMalloc((void **)&d_MeanGpp, size);
    err = cudaMalloc((void **)&d_DayGPP, size);

    err = cudaMalloc((void **)&d_p0, size);
    err = cudaMalloc((void **)&d_p1, size);
    err = cudaMalloc((void **)&d_p2, size);
    err = cudaMalloc((void **)&d_p3, size);
    err = cudaMalloc((void **)&d_p4, size);
    err = cudaMalloc((void **)&d_p5, size);
    // err = cudaMalloc((void **)&d_tiemsers, sizeof(double));
    err = cudaMalloc((void **)&d_doy, sizeof(double));

    err = cudaMalloc((void **)&d_Lai, size);

    err = cudaMalloc((void **)&d_Lat, size);
}

void GetDayData::GetDayDataGPU(std::map<std::string, std::deque<VarInfo *> *> TleafDeque,
                               std::deque<double *> *Output)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < TleafDeque.find("t2m")->second->size(); i += h_DaySize)
    {
        double *DayMean = (double *)malloc(size);

        double *h_DaySum = (double *)(TleafDeque.find("t2m")->second->at(i)->Data.front());
        err = cudaMemcpy(d_sum, h_DaySum, size, cudaMemcpyHostToDevice);

        for (int j = i + 1; j < i + h_DaySize; j++)
        {
            double *h_Day3 = (double *)(TleafDeque.find("t2m")->second->at(j)->Data.front());
            err = cudaMemcpy(d_Gpp, h_Day3, size, cudaMemcpyHostToDevice);
            accum_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_sum);
        }
        GetMean_C(blocksPerGrid, threadsPerBlock, d_sum, d_DaySize, d_MeanGpp);

        err = cudaMemcpy(DayMean, d_MeanGpp, size, cudaMemcpyDeviceToHost);

        Output->push_back(DayMean);
    }
}

void GetDayData::GetDayDataGPU(std::deque<double *> *DataDeque,
                               std::deque<double *> *Output)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < DataDeque->size(); i += h_DaySize)
    {
        double *DayMean = (double *)malloc(size);

        double *h_DaySum = DataDeque->at(i);
        err = cudaMemcpy(d_sum, h_DaySum, size, cudaMemcpyHostToDevice);

        for (int j = i + 1; j < i + h_DaySize; j++)
        {
            double *h_Day3 = DataDeque->at(j);
            err = cudaMemcpy(d_Gpp, h_Day3, size, cudaMemcpyHostToDevice);
            accum_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_sum);
        }
        GetMean_C(blocksPerGrid, threadsPerBlock, d_sum, d_DaySize, d_MeanGpp);

        err = cudaMemcpy(DayMean, d_MeanGpp, size, cudaMemcpyDeviceToHost);

        Output->push_back(DayMean);
    }
}

void GetDayData::GetYearDataGPU(std::deque<double *> *DataDeque,
                                std::deque<double *> *Output)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

    double *Year = (double *)malloc(size);

    double *h_YearSum = DataDeque->at(0);
    err = cudaMemcpy(d_sum, h_YearSum, size, cudaMemcpyHostToDevice);

    for (int j = 1; j < DataDeque->size(); j++)
    {
        double *h_Day = DataDeque->at(j);
        err = cudaMemcpy(d_Gpp, h_Day, size, cudaMemcpyHostToDevice);
        accum_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_sum);
    }
    err = cudaMemcpy(Year, d_sum, size, cudaMemcpyDeviceToHost);

    Output->push_back(Year);
}

void GetDayData::GetDayDataGPU(std::deque<double *> *GppDeque,
                               std::deque<InformStruct *> *OutLaiInputDeque,
                               std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque,
                               std::deque<double *> *Output)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

    err = cudaMemcpy(d_p0, OutLaiInputDeque->back()->LaiBand->p0, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p1, OutLaiInputDeque->back()->LaiBand->p1, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p2, OutLaiInputDeque->back()->LaiBand->p2, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p3, OutLaiInputDeque->back()->LaiBand->p3, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p4, OutLaiInputDeque->back()->LaiBand->p4, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p5, OutLaiInputDeque->back()->LaiBand->p5, size, cudaMemcpyHostToDevice);

    err = cudaMemcpy(d_Lat, OutLaiInputDeque->back()->LaiBand->Lat, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < GppDeque->size(); i += h_DaySize)
    {
        ///////////get day of year////////////
        double doy = ceil(i / h_DaySize) + 1;
        err = cudaMemcpy(d_doy, &doy, sizeof(double), cudaMemcpyHostToDevice);

        // ////////get day GPP/////////////////
        double *DayGPP = (double *)malloc(size);

        double *h_sum = GppDeque->at(i);
        err = cudaMemcpy(d_sum, h_sum, size, cudaMemcpyHostToDevice);

        for (int j = i + 1; j < i + h_DaySize; j++)
        {
            double *h_Gpp = GppDeque->at(j);
            err = cudaMemcpy(d_Gpp, h_Gpp, size, cudaMemcpyHostToDevice);
            accum_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_sum);
        }
        GetMean_C(blocksPerGrid, threadsPerBlock, d_sum, d_DaySize, d_MeanGpp);

        ////get day LAI//////////////////
        inputLai_C(blocksPerGrid, threadsPerBlock, d_p0, d_p1, d_p2, d_p3, d_p4, d_p5, d_doy, d_Lai);

        /////get day GPP///////////////
        GetDayData_C(blocksPerGrid, threadsPerBlock, d_MeanGpp, d_Lai, d_DayGPP);

        //////////result/////////////////////
        err = cudaMemcpy(DayGPP, d_DayGPP, size, cudaMemcpyDeviceToHost);

        Output->push_back(DayGPP);
    }
}

void GetDayData::Release()
{
    err = cudaFree(d_sum);
    err = cudaFree(d_Gpp);
    err = cudaFree(d_DaySize);
    err = cudaFree(d_MeanGpp);

    err = cudaFree(d_p0);
    err = cudaFree(d_p1);
    err = cudaFree(d_p2);
    err = cudaFree(d_p3);
    err = cudaFree(d_p4);
    err = cudaFree(d_p5);
    err = cudaFree(d_doy);

    err = cudaFree(d_Lai);

    err = cudaFree(d_Lat);
}