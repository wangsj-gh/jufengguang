#pragma once

#include "StructHeader.h"

extern "C" void inputLai_C(int blocksPerGrid, int threadsPerBlock, const double *p0, const double *p1, const double *p2,
                           const double *p3, const double *p4, const double *p5, const double *t, double *LaiData);

extern "C" void inputData_C(int blocksPerGrid, int threadsPerBlock, const double *Lon, const double *Lat, const double *Temp,
                            const double *Rad, const double *dewpoint, const double *LAI,
                            double *VPD, double *RH, double *Vcmax, double *Jmax);

extern "C" void SZA_kb_calculation_C(int blocksPerGrid, int threadsPerBlock, const double *Lat, const double *Lon, const double *doy, const double *timeSeries,
                                     double *miukb, double *G, double *kb);

extern "C" void rad_transfer_C(int blocksPerGrid, int threadsPerBlock, const double *LAI, const double *kb, const double *G,
                               double *taob, double *betab, double *taod, double *betad);

extern "C" void rad_partition_C(int blocksPerGrid, int threadsPerBlock, const double *Rad, const double *miukb,double *Rad_last,
                                double *Rad_direct, double *Rad_diffuse);

extern "C" void twoleaf_C(int blocksPerGrid, int threadsPerBlock, const double *LAI, const double *Rad_direct, const double *Rad_diffuse,
                          const double *taob, const double *betab, const double *taod, const double *betad, const double *kb, const double *clumping,
                          double *LAI_sun, double *LAI_shade, double *PPFD_sun, double *PPFD_shade);

extern "C" void PSN_C3_C(int blocksPerGrid, int threadsPerBlock, const double *VPD, const double *PPFD, const double *Vcmax,
                         const double *Jmax, const double *Tleaf, const double *RH, const char *gsmodel,
                         double *A, double *GS, double *Rd);

extern "C" void PSN_C4_C(int blocksPerGrid, int threadsPerBlock, const double *VPD, const double *PPFD, const double *Vcmax,
                         const double *Jmax, const double *Tleaf, const double *RH, const char *gsmodel,
                         double *A, double *GS, double *Rd);

extern "C" void total_PSN_C3(int blocksPerGrid, int threadsPerBlock, const double *A_sun, const double *GS_sun, const double *Rd_sun,
                             const double *A_shade, const double *GS_shade, const double *Rd_shade,
                             const double *LAI_sun, const double *LAI_shade,
                             double *total_A, double *total_GS, double *total_Rd);

extern "C" void total_PSN_C4(int blocksPerGrid, int threadsPerBlock, const double *A_sun, const double *GS_sun, const double *Rd_sun,
                             const double *A_shade, const double *GS_shade, const double *Rd_shade,
                             const double *LAI_sun, const double *LAI_shade,
                             double *total_A, double *total_GS, double *total_Rd);

extern "C" void Photosynthesis_C(int blocksPerGrid, int threadsPerBlock, const double *A_C3, const double *GS_C3, const double *Rd_C3,
                                 const double *A_C4, const double *GS_C4, const double *Rd_C4, const double *PercentC4,
                                 double *Photosynthesis_A, double *Photosynthesis_GS, double *Photosynthesis_Rd);

class PSN
{
public:
    void Init(std::deque<InformStruct *> *OutLaiInputDeque);
    void Inneed();
    bool PSNGPU(std::deque<InformStruct *> *OutLaiInputDeque,
                std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque,
                std::deque<double *> *ClumpIndexInputDeque,
                std::deque<double *> *PercentC4InputDeque,
                std::map<std::string, std::deque<VarInfo *> *> VarInfoMapInputDeque,
                std::deque<double *> *Output);
    void Release();
private:
    /// input data//////////////////////
    std::deque<InformStruct *> *OutLaiInputDeque;
    std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque;
    std::deque<double *> *ClumpIndexInputDeque;
    std::deque<double *> *PercentC4InputDeque;
    std::map<std::string, std::deque<VarInfo *> *> VarInfoMapInputDeque;

    // double *h_Pressure;
    double *h_Rad;
    double *h_dewpoint;
    double *h_Temp;

    double h_tiemsers;
    double h_doy;
    /// from host copy to device//////////////////////////
    double *d_Lon;
    double *d_Lat;

    double *d_p0;
    double *d_p1;
    double *d_p2;
    double *d_p3;
    double *d_p4;
    double *d_p5;

    double *d_ClumpIndex;
    double *d_PercentC4;

    // double *d_Pressure;
    double *d_Rad;
    double *d_Rad_last;
    double *d_dewpoint;
    double *d_Temp;

    double *d_tiemsers;
    double *d_doy;

    /// Variables participating in the operation///////////////////
    double *d_Lai;
    double *d_VPD;
    double *d_RH;
    double *d_Vcmax;
    double *d_Jmax;

    double *d_miukb;
    double *d_G;
    double *d_kb;

    double *d_taob;
    double *d_betab;
    double *d_taod;
    double *d_betad;

    double *d_Rad_direct;
    double *d_Rad_diffuse;

    double *d_LAI_sun;
    double *d_LAI_shade;
    double *d_PPFD_sun;
    double *d_PPFD_shade;

    double *d_A_sun_C3;
    double *d_GS_sun_C3;
    double *d_Rd_sun_C3;
    double *d_A_shade_C3;
    double *d_GS_shade_C3;
    double *d_Rd_shade_C3;
    double *d_A_total_C3;
    double *d_GS_total_C3;
    double *d_Rd_total_C3;

    double *d_A_sun_C4;
    double *d_GS_sun_C4;
    double *d_Rd_sun_C4;
    double *d_A_shade_C4;
    double *d_GS_shade_C4;
    double *d_Rd_shade_C4;
    double *d_A_total_C4;
    double *d_GS_total_C4;
    double *d_Rd_total_C4;

    double *d_Photosynthesis_A;
    double *d_Photosynthesis_GS;
    double *d_Photosynthesis_Rd;

    cudaError_t err;
    size_t size;
    int numElements;
};
