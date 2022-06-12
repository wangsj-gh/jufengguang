#pragma once

#include "StructHeader.h"
#include "GetDayData.h"

extern "C" void MR_Leaf_C(int blocksPerGrid, int threadsPerBlock,
                          const double *dayTleaf, const double *dayLAI, const double *landcover,
                          double *dayLeaf_Mr, double *dayLeaf_Mass, double *Leaf_Mass_Max);
extern "C" void MR_FineRoot_C(int blocksPerGrid, int threadsPerBlock,
                              const double *dayTleaf, const double *landcover, const double *dayLeaf_Mass,
                              double *dayFroot_Mr, double *annsum_mr);
extern "C" void yearGppLeafFineroot_C(int blocksPerGrid, int threadsPerBlock,
                                      const double *dayGpp, const double *dayLeaf_Mr, const double *dayFroot_Mr,
                                      double *yearGpp, double *yearLeaf_Mr, double *yearFroot_Mr);
extern "C" void MR_liveWood_C(int blocksPerGrid, int threadsPerBlock,
                              const double *Leaf_Mass_Max, const double *landcover, const double *annsum_mr,
                              double *yearLivewood_Mr);
extern "C" void Mrespiration_C(int blocksPerGrid, int threadsPerBlock,
                               const double *yearGpp, const double *yearLeaf_Mr,
                               const double *yearFroot_Mr, const double *yearLivewood_Mr,
                               double *NPP);
class Pespiration
{

public:
    void Init(std::deque<InformStruct *> *OutLaiInputDeque,
              std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque);
    void Inneed();

    void PespirationGPU(std::deque<InformStruct *> *OutLaiInputDeque,
                        std::deque<double *> *TleafMeanDeque,
                        std::deque<double *> *GppMeanDeque,
                        std::deque<double *> *LandCoverInputDeque,
                        std::deque<double *> *Output);

    void Release();

private:
    std::deque<InformStruct *> *OutLaiInputDeque;
    std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque;
    std::deque<double *> *TleafMeanDeque;
    std::deque<double *> *GppMeanDeque;
    std::deque<double *> *LandCoverInputDeque;
    int h_DaySize;

    //////////////MR_Leaf/////////////
    double *d_dayTleaf;
    double *d_dayLai;
    double *d_landcover;
    double *d_dayLeaf_Mr;
    double *d_dayLeaf_Mass;
    double *d_Leaf_Mass_Max;

    /// lai////
    double *d_p0;
    double *d_p1;
    double *d_p2;
    double *d_p3;
    double *d_p4;
    double *d_p5;
    double *d_daydoy;

    //////////////MR_FineRoot/////////////
    double *d_dayFroot_Mr;
    double *d_annsum_mr;

    //////////////yearGppLeafFineroot/////////////
    double *d_dayGpp;
    double *d_yearGpp;
    double *d_yearLeaf_Mr;
    double *d_yearFroot_Mr;

    //////////////MR_liveWood/////////////
    double *d_yearLivewood_Mr;

    //////////////Mrespiration/////////////
    double *d_NPP;

    cudaError_t err;
    size_t size;
    int numElements;
};