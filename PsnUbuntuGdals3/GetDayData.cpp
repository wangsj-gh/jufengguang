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

    double *d_tiemsers = NULL;
    double *d_doy = NULL;

    double *d_Lat = NULL;
    double *d_Lon = NULL;

    double *d_Sunrise = NULL;
    double *d_Sunset = NULL; 

    double *d_LocalTime = NULL;
    double *d_CorrectionFactor = NULL;

}

void GetDayData::Inneed()
{
    err = cudaMalloc((void **)&d_sum, size);
    err = cudaMalloc((void **)&d_Gpp, size);
    err = cudaMalloc((void **)&d_DaySize, size);
    err = cudaMalloc((void **)&d_MeanGpp, size);
    err = cudaMalloc((void **)&d_DayGPP, size);

    err = cudaMalloc((void **)&d_tiemsers, sizeof(double));
    err = cudaMalloc((void **)&d_doy, sizeof(double));

    err = cudaMalloc((void **)&d_Lat, size);
    err = cudaMalloc((void **)&d_Lon, size);

    err = cudaMalloc((void **)&d_Sunrise, size);
    err = cudaMalloc((void **)&d_Sunset, size);

    err = cudaMalloc((void **)&d_LocalTime, size);
    err = cudaMalloc((void **)&d_CorrectionFactor, size);

}

void GetDayData::GetDayDataGPU(std::map<std::string, std::deque<VarInfo *> *> TleafDeque,
                               std::deque<double *> *Output)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < TleafDeque.find("Tair_f_inst")->second->size(); i += h_DaySize)
    {
        double *DayMean = (double *)malloc(size);

        double *h_DaySum = (double *)(TleafDeque.find("Tair_f_inst")->second->at(i)->Data.front());
        err = cudaMemcpy(d_sum, h_DaySum, size, cudaMemcpyHostToDevice);

        for (int j = i + 1; j < i + h_DaySize; j++)
        {
            double *h_Day3 = (double *)(TleafDeque.find("Tair_f_inst")->second->at(j)->Data.front());
            err = cudaMemcpy(d_Gpp, h_Day3, size, cudaMemcpyHostToDevice);
            accum_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_sum);
        }

        err = cudaMemcpy(DayMean, d_MeanGpp, size, cudaMemcpyDeviceToHost);

        Output->push_back(DayMean);
    }
}

// void GetDayData::GetDayDataGPU(std::deque<double *> *DataDeque,
//                                std::deque<double *> *Output)
// {
//     // Launch the Vector Add CUDA Kernel
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
//     err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

//     for (int i = 0; i < DataDeque->size(); i += h_DaySize)
//     {
//         double *DayMean = (double *)malloc(size);

//         double *h_DaySum = DataDeque->at(i);
//         err = cudaMemcpy(d_sum, h_DaySum, size, cudaMemcpyHostToDevice);

//         for (int j = i + 1; j < i + h_DaySize; j++)
//         {
//             double *h_Day3 = DataDeque->at(j);
//             err = cudaMemcpy(d_Gpp, h_Day3, size, cudaMemcpyHostToDevice);
//             accum_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_sum);
//         }
//         GetMean_C(blocksPerGrid, threadsPerBlock, d_sum, d_DaySize, d_MeanGpp);

//         err = cudaMemcpy(DayMean, d_MeanGpp, size, cudaMemcpyDeviceToHost);

//         Output->push_back(DayMean);
//     }
// }

void GetDayData::GetYearDataGPU(std::deque<double *> *DataDeque,
                                std::deque<double *> *Output)
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    err = cudaMemcpy(d_DaySize, &h_DaySize, size, cudaMemcpyHostToDevice);

    double *Year = (double *)malloc(size);
    double *h_YearSum = (double *)malloc(size);
    memset(h_YearSum,0,size);

    err = cudaMemcpy(d_sum, h_YearSum, size, cudaMemcpyHostToDevice);

    for (int j = 0; j < DataDeque->size(); j++)
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

    err = cudaMemcpy(d_Lat, OutLaiInputDeque->back()->LaiBand->Lat, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_Lon, OutLaiInputDeque->back()->LaiBand->Lon, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < GppDeque->size(); i += h_DaySize)
    {
        ///////////get day of year////////////
        double doy = (double)i / h_DaySize + 1;
        err = cudaMemcpy(d_doy, &doy, sizeof(double), cudaMemcpyHostToDevice);

        ///////////get sunrise and sunset//////////////
        GetSunRiseSet_C(blocksPerGrid, threadsPerBlock, d_Lat, d_doy, d_Sunrise, d_Sunset);

        // ////////get day GPP/////////////////
        double *DayGPP = (double *)malloc(size);
        double *h_sum = (double *)malloc(size);
        memset(h_sum,0,size);
        // double *h_sum = GppDeque->at(i);
        err = cudaMemcpy(d_sum, h_sum, size, cudaMemcpyHostToDevice);
        for (int j = i ; j < i + h_DaySize; j++)
        {
            double *h_Gpp = GppDeque->at(j);
            err = cudaMemcpy(d_Gpp, h_Gpp, size, cudaMemcpyHostToDevice);

            ///////////get timesers////////////////
            int index = j % h_DaySize;
            double h_tiemsers = SpatiotemporalDeque->back()->timeSeries[index];
            err = cudaMemcpy(d_tiemsers, &h_tiemsers, sizeof(double), cudaMemcpyHostToDevice);

            //////////gte local time//////////////////////
            GetLocalTime_C(blocksPerGrid, threadsPerBlock, d_Lon, d_tiemsers, d_LocalTime);

            ///////////get CorrectionFactor//////////////
            GetCorrectionFactor_C(blocksPerGrid, threadsPerBlock, d_Sunrise, d_Sunset, d_LocalTime,d_DaySize, d_CorrectionFactor);

            GetDayData_C(blocksPerGrid, threadsPerBlock, d_Gpp, d_CorrectionFactor, d_DayGPP);

            accum_C(blocksPerGrid, threadsPerBlock, d_DayGPP, d_sum);

            // err = cudaMemcpy(DayGPP, d_CorrectionFactor, size, cudaMemcpyDeviceToHost);
            // Output->push_back(DayGPP);
        }
        //////////result/////////////////////
        err = cudaMemcpy(DayGPP, d_sum, size, cudaMemcpyDeviceToHost);
        Output->push_back(DayGPP);
    }
}

void GetDayData::Release()
{
    err = cudaFree(d_sum);
    err = cudaFree(d_Gpp);
    err = cudaFree(d_DaySize);
    err = cudaFree(d_MeanGpp);

    err = cudaFree(d_doy);

    err = cudaFree(d_tiemsers);
    err = cudaFree(d_Lat);
    err = cudaFree(d_Lon);

    err = cudaFree(d_Sunrise);
    err = cudaFree(d_Sunset);
    err = cudaFree(d_LocalTime);
    err = cudaFree(d_CorrectionFactor);
}