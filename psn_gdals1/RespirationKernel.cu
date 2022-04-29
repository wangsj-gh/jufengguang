/*The Respiration model*/
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "StructHeader.h"

using namespace std;
/*the unit is umol co2/m2/day*/

//////////////////////leaf////////////////////
/*
NAME                                  value  samlpe
Water Bodies                            0      WB
Evergreen Needleleaf Forests            1      ENF      use
Evergreen Broadleaf Forests             2      EBF      use
Deciduous Needleleaf Forests            3      DNF      use
Deciduous Broadleaf Forests             4      DBF      use
Mixed Forests                           5      MF       use
Closed Shrublands                       6      CShrub   use
Open Shrublands                         7      OShrub   use
Woody Savannas                          8      WSavanna use
Savannas                                9      Savanna  use
Grasslands                             10      Grass    use
Permanent Wetlands                     11      PW
Croplands                              12      Crop     use
Urban and Built-up Lands               13
Cropland/Natural Vegetation Mosaics    14
Non-Vegetated Lands                    15
*/

///////////////////////leaf//////////////////////
__global__ void MR_Leaf(const double *dayTleaf, const double *dayLAI, const double *landcover,
                        double *dayLeaf_Mr, double *dayLeaf_Mass, double *Leaf_Mass_Max)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // WB   ENF      EBF     DNF     DBF      MF     CShrub  OShrub  WSavanna  Savanna  Grass     PW    Crop
    double SLA[13] = {0, 14.1, 25.9, 15.5, 21.8, 21.5, 9.0, 11.5, 27.4, 27.1, 37.5, 0, 30.4};
    double Leaf_Mr_base[13] = {0, 0.00604, 0.00604, 0.00815, 0.00778, 0.00778, 0.00869, 0.00519, 0.00869, 0.00869, 0.0098, 0, 0.0098};

    double Q10_leaf = 3.22 - 0.046 * dayTleaf[i];
    int index = (int)landcover[i];

    dayLeaf_Mass[i] = dayLAI[i] / SLA[index];
    Leaf_Mass_Max[i] = max(dayLeaf_Mass[i], Leaf_Mass_Max[i]);

    dayLeaf_Mr[i] = dayLeaf_Mass[i] * Leaf_Mr_base[index] * pow(Q10_leaf, (dayTleaf[i] - 20) / 10);
}

////////////////////fine root////////////////////////
__global__ void MR_FineRoot(const double *dayTleaf, const double *landcover, const double *dayLeaf_Mass,
                            double *dayFroot_Mr, double *annsum_mr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // WB   ENF      EBF     DNF     DBF      MF     CShrub  OShrub  WSavanna  Savanna  Grass     PW    Crop
    double Froot_Mr_base[] = {0, 0.00519, 0.00519, 0.00519, 0.00519, 0.00519, 0.00519, 0.00519, 0.00519, 0.00519, 0.00819, 0, 0.00819};
    double Froot_Leaf_Ratio[] = {0, 1.2, 1.1, 1.7, 1.1, 1.1, 1.0, 1.3, 1.8, 1.8, 2.6, 0, 2.0};

    double Q10 = 2.0;
    int index = (int)landcover[i];

    double day_mr = pow(Q10, (dayTleaf[i] - 20) / 10);

    double Fine_Root_Mass = dayLeaf_Mass[i] * Froot_Leaf_Ratio[index];
    dayFroot_Mr[i] = Fine_Root_Mass * Froot_Mr_base[index] * day_mr;

    annsum_mr[i] = annsum_mr[i] + day_mr;
}

////////////////////live wood////////////////////////
__global__ void MR_liveWood(const double *Leaf_Mass_Max, const double *landcover, const double *annsum_mr,
                            double *yearLivewood_Mr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // WB   ENF      EBF     DNF     DBF      MF     CShrub  OShrub  WSavanna  Savanna  Grass     PW    Crop
    double Livewood_Mr_base[13] = {0, 0.00397, 0.00397, 0.00397, 0.00371, 0.00371, 0.00436, 0.00218, 0.00312, 0.001, 0, 0, 0};
    double Livewood_Leaf_Ratio[13] = {0, 0.182, 0.162, 0.165, 0.203, 0.203, 0.079, 0.04, 0.091, 0.051, 0, 0, 0};

    int index = (int)landcover[i];

    double Livewood_Mass = Leaf_Mass_Max[i] * Livewood_Leaf_Ratio[index];
    yearLivewood_Mr[i] = Livewood_Mass * Livewood_Mr_base[index] * annsum_mr[i];
}

//////////////covert day unit to year unit by sum of year data/////////////////
__global__ void yearGppLeafFineroot(const double *dayGpp, const double *dayLeaf_Mr, const double *dayFroot_Mr,
                                    double *yearGpp, double *yearLeaf_Mr, double *yearFroot_Mr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    yearGpp[i] = yearGpp[i] + dayGpp[i];
    yearLeaf_Mr[i] = yearLeaf_Mr[i] + dayLeaf_Mr[i];
    yearFroot_Mr[i] = yearFroot_Mr[i] + dayFroot_Mr[i];
}

//////////////////////respiration////////////////////
__global__ void Mrespiration(const double *yearGpp, const double *yearLeaf_Mr,
                             const double *yearFroot_Mr, const double *yearLivewood_Mr,
                             double *NPP)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double Rm = yearLeaf_Mr[i] + yearFroot_Mr[i] + yearLivewood_Mr[i];

    if (yearGpp[i] >= Rm)
    {
        NPP[i] = 0.8 * (yearGpp[i] - Rm);
    }
    else
    {
        NPP[i] = 0.0;
    }

    if (NPP[i] == 0)
    {
        NPP[i] = NAN;
    }
}

extern "C" void MR_Leaf_C(int blocksPerGrid, int threadsPerBlock,
                          const double *dayTleaf, const double *dayLAI, const double *landcover,
                          double *dayLeaf_Mr, double *dayLeaf_Mass, double *Leaf_Mass_Max)
{
    MR_Leaf<<<blocksPerGrid, threadsPerBlock>>>(dayTleaf, dayLAI, landcover,
                                                dayLeaf_Mr, dayLeaf_Mass, Leaf_Mass_Max);
}

extern "C" void MR_FineRoot_C(int blocksPerGrid, int threadsPerBlock,
                              const double *dayTleaf, const double *landcover, const double *dayLeaf_Mass,
                              double *dayFroot_Mr, double *annsum_mr)
{
    MR_FineRoot<<<blocksPerGrid, threadsPerBlock>>>(dayTleaf, landcover, dayLeaf_Mass,
                                                    dayFroot_Mr, annsum_mr);
}

extern "C" void MR_liveWood_C(int blocksPerGrid, int threadsPerBlock,
                              const double *Leaf_Mass_Max, const double *landcover, const double *annsum_mr,
                              double *yearLivewood_Mr)
{
    MR_liveWood<<<blocksPerGrid, threadsPerBlock>>>(Leaf_Mass_Max, landcover,
                                                    annsum_mr, yearLivewood_Mr);
}

extern "C" void yearGppLeafFineroot_C(int blocksPerGrid, int threadsPerBlock,
                                      const double *dayGpp, const double *dayLeaf_Mr, const double *dayFroot_Mr,
                                      double *yearGpp, double *yearLeaf_Mr, double *yearFroot_Mr)
{
    yearGppLeafFineroot<<<blocksPerGrid, threadsPerBlock>>>(dayGpp, dayLeaf_Mr, dayFroot_Mr,
                                                            yearGpp, yearLeaf_Mr, yearFroot_Mr);
}

extern "C" void Mrespiration_C(int blocksPerGrid, int threadsPerBlock,
                               const double *yearGpp, const double *yearLeaf_Mr,
                               const double *yearFroot_Mr, const double *yearLivewood_Mr,
                               double *NPP)
{
    Mrespiration<<<blocksPerGrid, threadsPerBlock>>>(yearGpp, yearLeaf_Mr, yearFroot_Mr, yearLivewood_Mr,
                                                     NPP);
}