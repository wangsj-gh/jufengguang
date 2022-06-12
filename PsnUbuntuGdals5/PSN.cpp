
#include "PSN.h"
#include <iostream>

using namespace std;

void PSN::Init(std::deque<InformStruct *> *OutLaiInputDeque)
{

    TifRasterStruct *p_raster = OutLaiInputDeque->back()->TifRaster;

    numElements = p_raster->RasterXSize * p_raster->RasterYSize;
    size = numElements * sizeof(double);

    //////////////GetDayData class//////////////
    double *d_Sunrise = NULL;
    double *d_Sunset = NULL; 

    double *d_LocalTime = NULL;
    double *d_CorrectionFactor = NULL;
    int *d_DaySize = NULL;
    //////////////PSN class//////////////
    double *d_Lon = NULL;
    double *d_Lat = NULL;
    double *d_Pressure = NULL;
    double *d_Rad = NULL;
    double *d_SH = NULL;
    double *d_Temp = NULL;
    double *d_ClumpIndex = NULL;
    double *d_PercentC4 = NULL;
    double *d_landcover = NULL;

    double *d_p0 = NULL;
    double *d_p1 = NULL;
    double *d_p2 = NULL;
    double *d_p3 = NULL;
    double *d_p4 = NULL;
    double *d_p5 = NULL;
    double *d_tiemsers = NULL;
    double *d_doy = NULL;

    double *d_Lai = NULL;
    double *d_VPD = NULL;
    double *d_RH = NULL;

    double *d_miukb = NULL;
    double *d_G = NULL;
    double *d_kb = NULL;

    double *d_taob = NULL;
    double *d_betab = NULL;
    double *d_taod = NULL;
    double *d_betad = NULL;

    double *d_Rad_direct = NULL;
    double *d_Rad_diffuse = NULL;

    double *d_LAI_sun = NULL;
    double *d_LAI_shade = NULL;
    double *d_PPFD_sun = NULL;
    double *d_PPFD_shade = NULL;

    double *d_SLA_sun = NULL;
    double *d_SLA_shade = NULL;

    double *d_Vcmax_sun = NULL;
    double *d_Vcmax_shade = NULL;
    double *d_Vcmax25 = NULL;

    double *d_Jmax_sun = NULL;
    double *d_Jmax_shade = NULL;

    double *d_A_sun_C3 = NULL;
    double *d_GS_sun_C3 = NULL;
    double *d_Rd_sun_C3 = NULL;
    double *d_A_shade_C3 = NULL;
    double *d_GS_shade_C3 = NULL;
    double *d_Rd_shade_C3 = NULL;
    double *d_A_total_C3 = NULL;
    double *d_GS_total_C3 = NULL;
    double *d_Rd_total_C3 = NULL;

    double *d_A_sun_C4 = NULL;
    double *d_GS_sun_C4 = NULL;
    double *d_Rd_sun_C4 = NULL;
    double *d_A_shade_C4 = NULL;
    double *d_GS_shade_C4 = NULL;
    double *d_Rd_shade_C4 = NULL;
    double *d_A_total_C4 = NULL;
    double *d_GS_total_C4 = NULL;
    double *d_Rd_total_C4 = NULL;

    double *d_Photosynthesis_A = NULL;
    double *d_Photosynthesis_GS = NULL;
    double *d_Photosynthesis_Rd = NULL;
}

void PSN::Inneed()
{
    //////////////GetDayData class//////////////
    err = cudaMalloc((void **)&d_Sunrise, size);
    err = cudaMalloc((void **)&d_Sunset, size);
    err = cudaMalloc((void **)&d_LocalTime, size);
    err = cudaMalloc((void **)&d_CorrectionFactor, size);
    err = cudaMalloc((void **)&d_DaySize, sizeof(int));

    //////////////PSN class//////////////
    err = cudaMalloc((void **)&d_Lon, size);
    err = cudaMalloc((void **)&d_Lat, size);

    err = cudaMalloc((void **)&d_p0, size);
    err = cudaMalloc((void **)&d_p1, size);
    err = cudaMalloc((void **)&d_p2, size);
    err = cudaMalloc((void **)&d_p3, size);
    err = cudaMalloc((void **)&d_p4, size);
    err = cudaMalloc((void **)&d_p5, size);

    err = cudaMalloc((void **)&d_ClumpIndex, size);
    err = cudaMalloc((void **)&d_PercentC4, size);
    err = cudaMalloc((void **)&d_landcover, size);

    err = cudaMalloc((void **)&d_Pressure, size);
    err = cudaMalloc((void **)&d_Rad, size);
    err = cudaMalloc((void **)&d_SH, size);
    err = cudaMalloc((void **)&d_Temp, size);

    err = cudaMalloc((void **)&d_tiemsers, sizeof(double));
    err = cudaMalloc((void **)&d_doy, sizeof(double));

    ////////////////////////get double logistic////////////////////////////////
    err = cudaMalloc((void **)&d_Lai, size);
    ///////////////////////get input data/////////////////////////////////////
    err = cudaMalloc((void **)&d_VPD, size);
    err = cudaMalloc((void **)&d_RH, size);

    ////////////////////////SZA_kb_calculation//////////////////////////////
    err = cudaMalloc((void **)&d_miukb, size);
    err = cudaMalloc((void **)&d_G, size);
    err = cudaMalloc((void **)&d_kb, size);

    ////////////////////////rad_transfer//////////////////////////////
    err = cudaMalloc((void **)&d_taob, size);
    err = cudaMalloc((void **)&d_betab, size);
    err = cudaMalloc((void **)&d_taod, size);
    err = cudaMalloc((void **)&d_betad, size);

    ////////////////////////rad_partition////////////////////////////////////////////////
    err = cudaMalloc((void **)&d_Rad_direct, size);
    err = cudaMalloc((void **)&d_Rad_diffuse, size);

    ///////////////////////twoleaf///////////////////////////////////////////
    err = cudaMalloc((void **)&d_LAI_sun, size);
    err = cudaMalloc((void **)&d_LAI_shade, size);
    err = cudaMalloc((void **)&d_PPFD_sun, size);
    err = cudaMalloc((void **)&d_PPFD_shade, size);

    ///////////////////////vcmaxjmax///////////////////////////////////////////
    err = cudaMalloc((void **)&d_SLA_sun, size);
    err = cudaMalloc((void **)&d_SLA_shade, size);
    err = cudaMalloc((void **)&d_Vcmax_sun, size);
    err = cudaMalloc((void **)&d_Vcmax_shade, size);
    err = cudaMalloc((void **)&d_Vcmax25, size);
    err = cudaMalloc((void **)&d_Jmax_sun, size);
    err = cudaMalloc((void **)&d_Jmax_shade, size);

    ///////////////////////PSN_C3////////////////////////////
    err = cudaMalloc((void **)&d_A_sun_C3, size);
    err = cudaMalloc((void **)&d_GS_sun_C3, size);
    err = cudaMalloc((void **)&d_Rd_sun_C3, size);
    err = cudaMalloc((void **)&d_A_shade_C3, size);
    err = cudaMalloc((void **)&d_GS_shade_C3, size);
    err = cudaMalloc((void **)&d_Rd_shade_C3, size);

    err = cudaMalloc((void **)&d_A_total_C3, size);
    err = cudaMalloc((void **)&d_GS_total_C3, size);
    err = cudaMalloc((void **)&d_Rd_total_C3, size);

    ///////////////////////PSN_C4////////////////////////////
    err = cudaMalloc((void **)&d_A_sun_C4, size);
    err = cudaMalloc((void **)&d_GS_sun_C4, size);
    err = cudaMalloc((void **)&d_Rd_sun_C4, size);
    err = cudaMalloc((void **)&d_A_shade_C4, size);
    err = cudaMalloc((void **)&d_GS_shade_C4, size);
    err = cudaMalloc((void **)&d_Rd_shade_C4, size);

    err = cudaMalloc((void **)&d_A_total_C4, size);
    err = cudaMalloc((void **)&d_GS_total_C4, size);
    err = cudaMalloc((void **)&d_Rd_total_C4, size);

    ////////////////////////Photosynthesis//////////////////////////
    err = cudaMalloc((void **)&d_Photosynthesis_A, size);
    err = cudaMalloc((void **)&d_Photosynthesis_GS, size);
    err = cudaMalloc((void **)&d_Photosynthesis_Rd, size);
}

bool PSN::PSNGPU(std::deque<InformStruct *> *OutLaiInputDeque,
                 std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque,
                 std::deque<double *> *ClumpIndexInputDeque,
                 std::deque<double *> *PercentC4InputDeque,
                 std::map<std::string, std::deque<VarInfo *> *> VarInfoMapInputDeque,
                 std::deque<double *> *LandCoverInputDeque,
                 std::deque<double *> *Output)

{
    // Error code to check return values for CUDA calls
    // cudaError_t cudastatus;
    // if(err!=cudaSuccess)
    // {
    //     std::cout<<" "<<cudaGetErrorString(cudastatus)<<std::endl;
    // }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    err = cudaMemcpy(d_Lon, OutLaiInputDeque->back()->LaiBand->Lon, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_Lat, OutLaiInputDeque->back()->LaiBand->Lat, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p0, OutLaiInputDeque->back()->LaiBand->p0, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p1, OutLaiInputDeque->back()->LaiBand->p1, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p2, OutLaiInputDeque->back()->LaiBand->p2, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p3, OutLaiInputDeque->back()->LaiBand->p3, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p4, OutLaiInputDeque->back()->LaiBand->p4, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p5, OutLaiInputDeque->back()->LaiBand->p5, size, cudaMemcpyHostToDevice);

    err = cudaMemcpy(d_ClumpIndex, ClumpIndexInputDeque->back(), size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_PercentC4, PercentC4InputDeque->back(), size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_landcover, LandCoverInputDeque->back(), size, cudaMemcpyHostToDevice);

    h_DaySize = SpatiotemporalDeque->back()->TimeSize;
    err = cudaMemcpy(d_DaySize, &h_DaySize, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < VarInfoMapInputDeque.find("Psurf_f_inst")->second->size(); i++)
    {
        double *result = (double *)malloc(size);

        h_Pressure = (double *)(VarInfoMapInputDeque.find("Psurf_f_inst")->second->at(i)->Data.front());
        h_Rad = (double *)(VarInfoMapInputDeque.find("SWdown_f_tavg")->second->at(i)->Data.front());
        h_SH = (double *)(VarInfoMapInputDeque.find("Qair_f_inst")->second->at(i)->Data.front());
        h_Temp = (double *)(VarInfoMapInputDeque.find("Tair_f_inst")->second->at(i)->Data.front());
        h_tiemsers = SpatiotemporalDeque->back()->timeSeries[i % SpatiotemporalDeque->back()->TimeSize];
        h_doy = floor(i / SpatiotemporalDeque->back()->TimeSize) + 1.0;

        err = cudaMemcpy(d_Pressure, h_Pressure, size, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_Rad, h_Rad, size, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_SH, h_SH, size, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_Temp, h_Temp, size, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_tiemsers, &h_tiemsers, sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_doy, &h_doy, sizeof(double), cudaMemcpyHostToDevice);

        //////////////////////get CorrectionFactor////////////////////////////////
        GetSunRiseSet_C(blocksPerGrid, threadsPerBlock,d_Lat,d_doy,d_Sunrise,d_Sunset);

        GetLocalTime_C(blocksPerGrid, threadsPerBlock,d_Lon,d_tiemsers,d_LocalTime);

        GetCorrectionFactor_C(blocksPerGrid, threadsPerBlock,d_Sunrise,d_Sunset,d_LocalTime,d_DaySize,d_CorrectionFactor);

        //////////////////////get double logistic////////////////////////////////

        inputLai_C(blocksPerGrid, threadsPerBlock, d_p0, d_p1, d_p2, d_p3, d_p4, d_p5, d_doy, d_Lai);

        inputData_C(blocksPerGrid, threadsPerBlock, d_Temp, d_SH, d_Pressure,d_VPD, d_RH);

        SZA_kb_calculation_C(blocksPerGrid, threadsPerBlock, d_Lat, d_Lon, d_doy, d_tiemsers, d_miukb, d_G, d_kb);

        rad_transfer_C(blocksPerGrid, threadsPerBlock, d_Lai, d_kb, d_G, d_taob, d_betab, d_taod, d_betad);

        rad_partition_C(blocksPerGrid, threadsPerBlock, d_Rad, d_miukb,d_CorrectionFactor, d_Rad_direct, d_Rad_diffuse);

        twoleaf_C(blocksPerGrid, threadsPerBlock, d_Lai, d_Rad_direct, d_Rad_diffuse, d_taob, d_betab, d_taod, d_betad, d_kb, d_ClumpIndex,
                  d_LAI_sun, d_LAI_shade, d_PPFD_sun, d_PPFD_shade);

        GetSLA_C(blocksPerGrid, threadsPerBlock, d_LAI_sun, d_LAI_shade,d_Lai,d_landcover,d_SLA_sun, d_SLA_shade);

        GetVcmaxJmax_C(blocksPerGrid, threadsPerBlock, d_SLA_sun,d_Temp, d_landcover,d_Vcmax_sun, d_Jmax_sun, d_Vcmax25);

        GetVcmaxJmax_C(blocksPerGrid, threadsPerBlock, d_SLA_shade,d_Temp, d_landcover,d_Vcmax_shade, d_Jmax_shade, d_Vcmax25);
        
        ///////////////////////////C3////////////////////////////////////
        PSN_C3_C(blocksPerGrid, threadsPerBlock, d_VPD, d_PPFD_sun, d_Vcmax_sun, d_Jmax_sun, d_Temp, d_RH, "BBL",
                 d_A_sun_C3, d_GS_sun_C3, d_Rd_sun_C3);
        PSN_C3_C(blocksPerGrid, threadsPerBlock, d_VPD, d_PPFD_shade, d_Vcmax_shade, d_Jmax_shade, d_Temp, d_RH, "BBL",
                 d_A_shade_C3, d_GS_shade_C3, d_Rd_shade_C3);

        total_PSN_C3(blocksPerGrid, threadsPerBlock, d_A_sun_C3, d_GS_sun_C3, d_Rd_sun_C3,
                     d_A_shade_C3, d_GS_shade_C3, d_Rd_shade_C3, d_LAI_sun, d_LAI_shade,
                     d_A_total_C3, d_GS_total_C3, d_Rd_total_C3);

        //////////////////////////C4////////////////////////////////////////////////
        PSN_C4_C(blocksPerGrid, threadsPerBlock, d_VPD, d_PPFD_sun, d_Vcmax25, d_Temp, d_RH, "BBL",
                 d_A_sun_C4, d_GS_sun_C4, d_Rd_sun_C4);
        PSN_C4_C(blocksPerGrid, threadsPerBlock, d_VPD, d_PPFD_shade, d_Vcmax25, d_Temp, d_RH, "BBL",
                 d_A_shade_C4, d_GS_shade_C4, d_Rd_shade_C4);

        total_PSN_C4(blocksPerGrid, threadsPerBlock, d_A_sun_C4, d_GS_sun_C4, d_Rd_sun_C4,
                     d_A_shade_C4, d_GS_shade_C4, d_Rd_shade_C4, d_LAI_sun, d_LAI_shade,
                     d_A_total_C4, d_GS_total_C4, d_Rd_total_C4);

        ///////////////////////Photosynthesis//////////////////////////////////////
        Photosynthesis_C(blocksPerGrid, threadsPerBlock, d_A_total_C3, d_GS_total_C3, d_Rd_total_C3,
                         d_A_total_C4, d_GS_total_C4, d_Rd_total_C4, d_PercentC4,
                         d_Photosynthesis_A, d_Photosynthesis_GS, d_Photosynthesis_Rd);

        ////////////////////output//////////////////////
        err = cudaMemcpy(result, d_Photosynthesis_A, size, cudaMemcpyDeviceToHost);

        Output->push_back(result);

        err = cudaGetLastError();
    }
}

void PSN::Release()
{
    //////////////GetDayData class//////////////
    err = cudaFree(d_Sunrise);
    err = cudaFree(d_Sunset);
    err = cudaFree(d_LocalTime);
    err = cudaFree(d_CorrectionFactor);
    err = cudaFree(d_DaySize);

    //////////////PSN class//////////////
    err = cudaFree(d_p0);
    err = cudaFree(d_p1);
    err = cudaFree(d_p2);
    err = cudaFree(d_p3);
    err = cudaFree(d_p4);
    err = cudaFree(d_p5);
    err = cudaFree(d_doy);
    err = cudaFree(d_Lai);

    err = cudaFree(d_Lon);
    err = cudaFree(d_Lat);
    err = cudaFree(d_Temp);
    err = cudaFree(d_Rad);
    err = cudaFree(d_SH);
    err = cudaFree(d_Pressure);
    err = cudaFree(d_ClumpIndex);
    err = cudaFree(d_PercentC4);

    err = cudaFree(d_VPD);
    err = cudaFree(d_RH);

    err = cudaFree(d_tiemsers);
    err = cudaFree(d_miukb);
    err = cudaFree(d_G);
    err = cudaFree(d_kb);

    err = cudaFree(d_taob);
    err = cudaFree(d_betab);
    err = cudaFree(d_taod);
    err = cudaFree(d_betad);

    err = cudaFree(d_Rad);
    err = cudaFree(d_Rad_direct);
    err = cudaFree(d_Rad_diffuse);

    err = cudaFree(d_LAI_sun);
    err = cudaFree(d_LAI_shade);
    err = cudaFree(d_PPFD_sun);
    err = cudaFree(d_PPFD_shade);

    err = cudaFree(d_SLA_sun);
    err = cudaFree(d_SLA_shade);
    err = cudaFree(d_Vcmax_sun);
    err = cudaFree(d_Vcmax_shade);
    err = cudaFree(d_Vcmax25);
    
    err = cudaFree(d_Jmax_sun);
    err = cudaFree(d_Jmax_shade);

    err = cudaFree(d_A_sun_C3);
    err = cudaFree(d_GS_sun_C3);
    err = cudaFree(d_Rd_sun_C3);
    err = cudaFree(d_A_shade_C3);
    err = cudaFree(d_GS_shade_C3);
    err = cudaFree(d_Rd_shade_C3);
    err = cudaFree(d_A_total_C3);
    err = cudaFree(d_GS_total_C3);
    err = cudaFree(d_Rd_total_C3);

    err = cudaFree(d_A_sun_C4);
    err = cudaFree(d_GS_sun_C4);
    err = cudaFree(d_Rd_sun_C4);
    err = cudaFree(d_A_shade_C4);
    err = cudaFree(d_GS_shade_C4);
    err = cudaFree(d_Rd_shade_C4);
    err = cudaFree(d_A_total_C4);
    err = cudaFree(d_GS_total_C4);
    err = cudaFree(d_Rd_total_C4);

    err = cudaFree(d_Photosynthesis_A);
    err = cudaFree(d_Photosynthesis_GS);
    err = cudaFree(d_Photosynthesis_Rd);
}
