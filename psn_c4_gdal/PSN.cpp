
#include "PSN.h"
#include <iostream>

using namespace std;

void PSN::Init(TifRasterStruct* p_RasterDataSet)
{
    err = cudaSuccess;
    numElements = p_RasterDataSet->RasterXSize * p_RasterDataSet->RasterYSize;
    size = numElements * sizeof(double);

    double* d_Lon = NULL;
    double* d_Lat = NULL;
    double* d_Pressure = NULL;
    double* d_Rad = NULL;
    double* d_SH = NULL;
    double* d_Temp = NULL;
    double* d_p0 = NULL;
    double* d_p1 = NULL;
    double* d_p2 = NULL;
    double* d_p3 = NULL;
    double* d_p4 = NULL;
    double* d_p5 = NULL;
    double* d_tiemsers = NULL;
    double* d_doy = NULL;

    double* d_Lai = NULL;
    double* d_VPD = NULL;
    double* d_RH = NULL;
    double* d_Vcmax = NULL;
    double* d_Jmax = NULL;

    double* d_miukb = NULL;
    double* d_G = NULL;
    double* d_kb = NULL;

    double* d_taob = NULL;
    double* d_betab = NULL;
    double* d_taod = NULL;
    double* d_betad = NULL;

    double* d_Rad_direct = NULL;
    double* d_Rad_diffuse = NULL;

    double* d_LAI_sun = NULL;
    double* d_LAI_shade = NULL;
    double* d_PPFD_sun = NULL;
    double* d_PPFD_shade = NULL;

    double* d_A_sun = NULL;
    double* d_GS_sun = NULL;
    double* d_Rd_sun = NULL;
    double* d_A_shade = NULL;
    double* d_GS_shade = NULL;
    double* d_Rd_shade = NULL;

    double* d_A_total = NULL;
    double* d_GS_total = NULL;
    double* d_Rd_total = NULL;
}

void PSN::Inneed()
{
    err = cudaMalloc((void**)&d_Lon, size);
    err = cudaMalloc((void**)&d_Lat, size);
    err = cudaMalloc((void**)&d_Pressure, size);
    err = cudaMalloc((void**)&d_Rad, size);
    err = cudaMalloc((void**)&d_SH, size);
    err = cudaMalloc((void**)&d_Temp, size);
    err = cudaMalloc((void**)&d_p0, size);
    err = cudaMalloc((void**)&d_p1, size);
    err = cudaMalloc((void**)&d_p2, size);
    err = cudaMalloc((void**)&d_p3, size);
    err = cudaMalloc((void**)&d_p4, size);
    err = cudaMalloc((void**)&d_p5, size);
    err = cudaMalloc((void**)&d_tiemsers, sizeof(double));
    err = cudaMalloc((void**)&d_doy, sizeof(double));

    ////////////////////////get double logistic////////////////////////////////
    err = cudaMalloc((void**)&d_Lai, size);
    ///////////////////////get input data/////////////////////////////////////
    err = cudaMalloc((void**)&d_VPD, size);
    err = cudaMalloc((void**)&d_RH, size);
    err = cudaMalloc((void**)&d_Vcmax, size);
    err = cudaMalloc((void**)&d_Jmax, size);

    ////////////////////////SZA_kb_calculation//////////////////////////////
    err = cudaMalloc((void**)&d_miukb, size);
    err = cudaMalloc((void**)&d_G, size);
    err = cudaMalloc((void**)&d_kb, size);

    ////////////////////////rad_transfer//////////////////////////////
    err = cudaMalloc((void**)&d_taob, size);
    err = cudaMalloc((void**)&d_betab, size);
    err = cudaMalloc((void**)&d_taod, size);
    err = cudaMalloc((void**)&d_betad, size);

    ////////////////////////rad_partition////////////////////////////////////////////////
    err = cudaMalloc((void**)&d_Rad_direct, size);
    err = cudaMalloc((void**)&d_Rad_diffuse, size);

    ///////////////////////twoleaf///////////////////////////////////////////
    err = cudaMalloc((void**)&d_LAI_sun, size);
    err = cudaMalloc((void**)&d_LAI_shade, size);
    err = cudaMalloc((void**)&d_PPFD_sun, size);
    err = cudaMalloc((void**)&d_PPFD_shade, size);

    ///////////////////////PSN_C3////////////////////////////
    err = cudaMalloc((void**)&d_A_sun, size);
    err = cudaMalloc((void**)&d_GS_sun, size);
    err = cudaMalloc((void**)&d_Rd_sun, size);
    err = cudaMalloc((void**)&d_A_shade, size);
    err = cudaMalloc((void**)&d_GS_shade, size);
    err = cudaMalloc((void**)&d_Rd_shade, size);

    ////////////////////////total_PSN//////////////////////////
    err = cudaMalloc((void**)&d_A_total, size);
    err = cudaMalloc((void**)&d_GS_total, size);
    err = cudaMalloc((void**)&d_Rd_total, size);
}

bool PSN::PSNcomputeGPU(TifRasterStruct* p_RasterDataSet, double* h_Lat, double* h_Lon,
    double* h_Pressure, double* h_Rad, double* h_SH, double* h_Temp,
    double* h_p0, double* h_p1, double* h_p2, double* h_p3, double* h_p4, double* h_p5,
    double h_tiemsers, double h_doy, double* Output)
{
    // Error code to check return values for CUDA calls
    /*cudaError_t err = cudaSuccess;*/
    err = cudaMemcpy(d_Lon, h_Lon, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_Lat, h_Lat, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_Pressure, h_Pressure, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_Rad, h_Rad, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_SH, h_SH, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_Temp, h_Temp, size, cudaMemcpyHostToDevice);

    err = cudaMemcpy(d_p0, h_p0, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p1, h_p1, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p2, h_p2, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p3, h_p3, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p4, h_p4, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p5, h_p5, size, cudaMemcpyHostToDevice);

    err = cudaMemcpy(d_tiemsers, &h_tiemsers, sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_doy, &h_doy, sizeof(double), cudaMemcpyHostToDevice);

    // Verify that allocations succeeded
    if (h_Lon == NULL || h_Lat == NULL || h_Temp == NULL || h_Rad == NULL || h_SH == NULL || h_Pressure == NULL)
    {
        cout << "Failed to allocate host vectors!\n" << endl;
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    ////////////////////////get double logistic////////////////////////////////

    inputLai_C(blocksPerGrid, threadsPerBlock, d_p0, d_p1, d_p2, d_p3, d_p4, d_p5, d_doy, d_Lai);

    inputData_C(blocksPerGrid, threadsPerBlock, d_Lon, d_Lat, d_Temp, d_Rad, d_SH, d_Pressure,
        d_Lai, d_VPD, d_RH, d_Vcmax, d_Jmax);
    SZA_kb_calculation_C(blocksPerGrid, threadsPerBlock, d_Lat, d_Lon, d_doy, d_tiemsers, d_miukb, d_G, d_kb);

    rad_transfer_C(blocksPerGrid, threadsPerBlock, d_Lai, d_kb, d_G, d_taob, d_betab, d_taod, d_betad);

    rad_partition_C(blocksPerGrid, threadsPerBlock, d_Rad, d_miukb, d_Rad_direct, d_Rad_diffuse);

    twoleaf_C(blocksPerGrid, threadsPerBlock, d_Lai, d_Rad_direct, d_Rad_diffuse, d_taob, d_betab, d_taod, d_betad, d_kb,
        d_LAI_sun, d_LAI_shade, d_PPFD_sun, d_PPFD_shade);

    PSN_C4_C(blocksPerGrid, threadsPerBlock, d_VPD, d_PPFD_sun, d_Vcmax, d_Jmax, d_Temp, d_RH, "BBL",
        d_A_sun, d_GS_sun, d_Rd_sun);
    PSN_C4_C(blocksPerGrid, threadsPerBlock, d_VPD, d_PPFD_shade, d_Vcmax, d_Jmax, d_Temp, d_RH, "BBL",
        d_A_shade, d_GS_shade, d_Rd_shade);

    total_PSN_C4(blocksPerGrid, threadsPerBlock, d_A_sun, d_GS_sun, d_Rd_sun,
        d_A_shade, d_GS_shade, d_Rd_shade, d_LAI_sun, d_LAI_shade,
        d_A_total, d_GS_total, d_Rd_total);

    err = cudaMemcpy(Output, d_A_total, size, cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    
}

void PSN::Release()
{
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
    err = cudaFree(d_VPD);
    err = cudaFree(d_RH);
    err = cudaFree(d_Vcmax);
    err = cudaFree(d_Jmax);


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

    err = cudaFree(d_A_sun);
    err = cudaFree(d_GS_sun);
    err = cudaFree(d_Rd_sun);
    err = cudaFree(d_A_shade);
    err = cudaFree(d_GS_shade);
    err = cudaFree(d_Rd_shade);

    err = cudaFree(d_A_total);
    err = cudaFree(d_GS_total);
    err = cudaFree(d_Rd_total);
}
