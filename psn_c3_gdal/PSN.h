#pragma once

// #include "PSN.cuh"
#include "StructHeader.h"

extern "C" void inputLai_C(int blocksPerGrid, int threadsPerBlock, const double* p0, const double* p1, const double* p2,
    const double* p3, const double* p4, const double* p5, const double* t, double* LaiData);
extern "C" void inputData_C(int blocksPerGrid, int threadsPerBlock, const double* Lon, const double* Lat, const double* Temp,
    const double* Rad, const double* SH, const double* Pressure, const double* LAI,
    double* VPD, double* RH, double* Vcmax, double* Jmax);
extern "C" void SZA_kb_calculation_C(int blocksPerGrid, int threadsPerBlock, const double* Lat, const double* Lon, const double* doy, const double* timeSeries,
    double* miukb, double* G, double* kb);
extern "C" void rad_transfer_C(int blocksPerGrid, int threadsPerBlock, const double* LAI, const double* kb, const double* G,
    double* taob, double* betab, double* taod, double* betad);
extern "C" void rad_partition_C(int blocksPerGrid, int threadsPerBlock, const double* Rad, const double* miukb,
    double* Rad_direct, double* Rad_diffuse);
extern "C" void twoleaf_C(int blocksPerGrid, int threadsPerBlock, const double* LAI, const double* Rad_direct, const double* Rad_diffuse,
    const double* taob, const double* betab, const double* taod, const double* betad, const double* kb,
    double* LAI_sun, double* LAI_shade, double* PPFD_sun, double* PPFD_shade);
extern "C" void PSN_C3_C(int blocksPerGrid, int threadsPerBlock, const double* VPD, const double* PPFD, const double* Vcmax,
    const double* Jmax, const double* Tleaf, const double* RH, const char* gsmodel,
    double* A, double* GS, double* Rd);
extern "C" void total_PSN_C3 (int blocksPerGrid, int threadsPerBlock, const double* A_sun, const double* GS_sun, const double* Rd_sun,
    const double* A_shade, const double* GS_shade, const double* Rd_shade,
    const double* LAI_sun, const double* LAI_shade,
    double* total_A, double* total_GS, double* total_Rd);



class PSN
{
public:
    void Init(TifRasterStruct* p_RasterDataSet);
    void Inneed();
    void Release();
    bool PSNcomputeGPU(TifRasterStruct* p_RasterDataSet, double* h_Lat, double* h_Lon,
        double* h_Pressure, double* h_Rad, double* h_SH, double* h_Temp,
        double* h_p0, double* h_p1, double* h_p2, double* h_p3, double* h_p4, double* h_p5,
        double h_tiemsers, double h_doy, 
        double* Output1,double* Output2,double* Output3,double* Output4,double* Output5,double* Output6,
        double* Output7,double* Output8,double* Output9,double* Output10);
private:
    /// input data//////////////////////
    TifRasterStruct* p_RasterDataSet;
    double* h_Lat;
    double* h_Lon;
    double* h_Pressure;
    double* h_Rad;
    double* h_SH;
    double* h_Temp;
    double* h_p0;
    double* h_p1;
    double* h_p2;
    double* h_p3;
    double* h_p4;
    double* h_p5;
    double h_tiemsers;
    double h_doy;
    double* Output;
   
    ///from host copy to device//////////////////////////
    double* d_Lon;
    double* d_Lat;
    double* d_Pressure;
    double* d_Rad;
    double* d_SH;
    double* d_Temp;
    double* d_p0;
    double* d_p1;
    double* d_p2;
    double* d_p3;
    double* d_p4;
    double* d_p5;
    double* d_tiemsers;
    double* d_doy;

    /// Variables participating in the operation///////////////////
    double* d_Lai;
    double* d_VPD;
    double* d_RH;
    double* d_Vcmax;
    double* d_Jmax;

    double* d_miukb;
    double* d_G;
    double* d_kb;

    double* d_taob;
    double* d_betab;
    double* d_taod;
    double* d_betad;

    double* d_Rad_direct;
    double* d_Rad_diffuse;

    double* d_LAI_sun;
    double* d_LAI_shade;
    double* d_PPFD_sun;
    double* d_PPFD_shade;

    double* d_A_sun;
    double* d_GS_sun;
    double* d_Rd_sun;
    double* d_A_shade;
    double* d_GS_shade;
    double* d_Rd_shade;

    double* d_A_total;
    double* d_GS_total;
    double* d_Rd_total;

    cudaError_t err;
    size_t size;
    int numElements;
};



