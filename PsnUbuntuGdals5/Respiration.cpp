#include "Respiration.h"
#include "PSN.h"
#include "string.h"
#include <iostream>

using namespace std;

void Pespiration::Init(std::deque<InformStruct *> *OutLaiInputDeque,
                       std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque)
{
    TifRasterStruct *p_raster = OutLaiInputDeque->back()->TifRaster;
    numElements = p_raster->RasterXSize * p_raster->RasterYSize;
    size = numElements * sizeof(double);
    h_DaySize = SpatiotemporalDeque->back()->TimeSize;

    //////////////MR_Leaf/////////////
    double *d_dayTleaf = NULL;
    double *d_dayLai = NULL;
    double *d_landcover = NULL;
    double *d_dayLeaf_Mr = NULL;
    double *d_dayLeaf_Mass = NULL;
    double *d_Leaf_Mass_Max = NULL;

    /// lai////
    double *d_p0 = NULL;
    double *d_p1 = NULL;
    double *d_p2 = NULL;
    double *d_p3 = NULL;
    double *d_p4 = NULL;
    double *d_p5 = NULL;
    double *d_daydoy = NULL;

    //////////////MR_FineRoot/////////////
    double *d_dayFroot_Mr = NULL;
    double *d_annsum_mr = NULL;

    //////////////yearGppLeafFineroot/////////////
    double *d_dayGpp = NULL;
    double *d_yearGpp = NULL;
    double *d_yearLeaf_Mr = NULL;
    double *d_yearFroot_Mr = NULL;

    //////////////MR_liveWood/////////////
    double *d_yearLivewood_Mr = NULL;

    //////////////Mrespiration/////////////
    double *d_NPP = NULL;
}

void Pespiration::Inneed()
{
    //////////////MR_Leaf/////////////
    err = cudaMalloc((void **)&d_dayTleaf, size);
    err = cudaMalloc((void **)&d_dayLai, size);
    err = cudaMalloc((void **)&d_landcover, size);
    err = cudaMalloc((void **)&d_dayLeaf_Mr, size);
    err = cudaMalloc((void **)&d_dayLeaf_Mass, size);
    err = cudaMalloc((void **)&d_Leaf_Mass_Max, size);

    /// lai////
    err = cudaMalloc((void **)&d_p0, size);
    err = cudaMalloc((void **)&d_p1, size);
    err = cudaMalloc((void **)&d_p2, size);
    err = cudaMalloc((void **)&d_p3, size);
    err = cudaMalloc((void **)&d_p4, size);
    err = cudaMalloc((void **)&d_p5, size);
    err = cudaMalloc((void **)&d_daydoy, sizeof(double));

    //////////////MR_FineRoot/////////////
    err = cudaMalloc((void **)&d_dayFroot_Mr, size);
    err = cudaMalloc((void **)&d_annsum_mr, size);

    //////////////yearGppLeafFineroot/////////////
    err = cudaMalloc((void **)&d_dayGpp, size);
    err = cudaMalloc((void **)&d_yearGpp, size);
    err = cudaMalloc((void **)&d_yearLeaf_Mr, size);
    err = cudaMalloc((void **)&d_yearFroot_Mr, size);

    //////////////MR_liveWood/////////////
    err = cudaMalloc((void **)&d_yearLivewood_Mr, size);

    //////////////Mrespiration/////////////
    err = cudaMalloc((void **)&d_NPP, size);
}

void Pespiration::PespirationGPU(std::deque<InformStruct *> *OutLaiInputDeque,
                                 std::deque<double *> *TleafMeanDeque,
                                 std::deque<double *> *GppMeanDeque,
                                 std::deque<double *> *LandCoverInputDeque,
                                 std::deque<double *> *Output)
{
    // std::cout << "OutLaiInputDeque:" << OutLaiInputDeque->size() << std::endl;
    // std::cout << "TleafMeanDeque:" << TleafMeanDeque->size() << std::endl;
    // std::cout << "GppMeanDeque:" << GppMeanDeque->size() << std::endl;
    // std::cout << "LandCoverInputDeque:" << LandCoverInputDeque->size() << std::endl;
    // Launch the Vector Add CUDA Kernel

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    err = cudaMemcpy(d_p0, OutLaiInputDeque->back()->LaiBand->p0, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p1, OutLaiInputDeque->back()->LaiBand->p1, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p2, OutLaiInputDeque->back()->LaiBand->p2, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p3, OutLaiInputDeque->back()->LaiBand->p3, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p4, OutLaiInputDeque->back()->LaiBand->p4, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_p5, OutLaiInputDeque->back()->LaiBand->p5, size, cudaMemcpyHostToDevice);

    err = cudaMemcpy(d_landcover, LandCoverInputDeque->back(), size, cudaMemcpyHostToDevice);

    /////Leaf_Mass_Max is the maximum value of year////////
    cudaMemset(d_Leaf_Mass_Max,0,size);

    /////annsum_mr is the sum value of year////////
    cudaMemset(d_annsum_mr,0,size);

    //////yearGpp,yearLeaf_Mr,yearFroot_Mr is the sum value of year//////
    cudaMemset(d_yearGpp,0,size);
    cudaMemset(d_yearLeaf_Mr,0,size);
    cudaMemset(d_yearFroot_Mr,0,size);

    ///////////////////////////////////
    double *result = (double *)malloc(size);
    for (int i = 0; i < GppMeanDeque->size(); i++)
    {
        // std::cout << "i finish" << i << std::endl;
        double *h_dayGpp = GppMeanDeque->at(i);
        double *h_dayTleaf = TleafMeanDeque->at(i);
        int h_daydoy = i + 1;
        err = cudaMemcpy(d_dayGpp, h_dayGpp, size, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_dayTleaf, h_dayTleaf, size, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_daydoy, &h_daydoy, sizeof(int), cudaMemcpyHostToDevice);

        inputLai_C(blocksPerGrid, threadsPerBlock,
                   d_p0, d_p1, d_p2, d_p3, d_p4, d_p5, d_daydoy, d_dayLai);

        MR_Leaf_C(blocksPerGrid, threadsPerBlock,
                  d_dayTleaf, d_dayLai, d_landcover,
                  d_dayLeaf_Mr, d_dayLeaf_Mass, d_Leaf_Mass_Max);

        MR_FineRoot_C(blocksPerGrid, threadsPerBlock,
                      d_dayTleaf, d_landcover, d_dayLeaf_Mass,
                      d_dayFroot_Mr, d_annsum_mr);

        yearGppLeafFineroot_C(blocksPerGrid, threadsPerBlock,
                              d_dayGpp, d_dayLeaf_Mr, d_dayFroot_Mr,
                              d_yearGpp, d_yearLeaf_Mr, d_yearFroot_Mr);
    }
    MR_liveWood_C(blocksPerGrid, threadsPerBlock,
                  d_Leaf_Mass_Max, d_landcover, d_annsum_mr,
                  d_yearLivewood_Mr);

    Mrespiration_C(blocksPerGrid, threadsPerBlock,
                   d_yearGpp, d_yearLeaf_Mr, d_yearFroot_Mr, d_yearLivewood_Mr,
                   d_NPP);

    err = cudaMemcpy(result, d_NPP, size, cudaMemcpyDeviceToHost);
    Output->push_back(result);

    std::cout << "end finish" << std::endl;
}

void Pespiration::Release()
{
    //////////////MR_Leaf/////////////
    err = cudaFree(d_dayTleaf);
    err = cudaFree(d_dayLai);
    err = cudaFree(d_landcover);
    err = cudaFree(d_dayLeaf_Mr);
    err = cudaFree(d_dayLeaf_Mass);
    err = cudaFree(d_Leaf_Mass_Max);

    /// lai////
    err = cudaFree(d_p0);
    err = cudaFree(d_p1);
    err = cudaFree(d_p2);
    err = cudaFree(d_p3);
    err = cudaFree(d_p4);
    err = cudaFree(d_p5);
    err = cudaFree(d_daydoy);

    //////////////MR_FineRoot/////////////
    err = cudaFree(d_dayFroot_Mr);
    err = cudaFree(d_annsum_mr);

    //////////////yearGppLeafFineroot/////////////
    err = cudaFree(d_dayGpp);
    err = cudaFree(d_yearGpp);
    err = cudaFree(d_yearLeaf_Mr);
    err = cudaFree(d_yearFroot_Mr);

    //////////////MR_liveWood/////////////
    err = cudaFree(d_yearLivewood_Mr);

    //////////////Mrespiration/////////////
    err = cudaFree(d_NPP);
}