#include <iostream>
#include <math.h>
#include <string.h>
using namespace std;




int main(void)
{
double CNr = 42; //gC/gN. C:N of leaves
double Flnr = 0.04; //gN in Rubisco / gN. Fraction of leaf N in Rubisco
double SLA_total = 0.012; //m2/gC. 
double SLA_ratio = 2; //ratio of shaded SLA : sunlit SLA  #BGC

double Tleaf = 20;
double LAI = 2.5;
double LAI_sun = 1.5;
double LAI_shade = 1;

double SLA_sun = (LAI_sun + LAI_shade / SLA_ratio) / (LAI / SLA_total);
double SLA_shade = SLA_sun * SLA_ratio;

std::cout<<"SLA_sun:"<<SLA_sun<<"  "<<"SLA_shade:"<<SLA_shade<<std::endl;
//CLM4.5
double Rgas = 8.314;
double Frn = 7.16; //gRubisco/gN in Rubisco.
double ACT25 = 60; //umolCO2/gRubisco/s
double Q10_ACT = 2.4;
double dha_vcmax =65330;
double dha_jmax =43540;
double ds_vcmax = 485;
double ds_jmax = 495;
double dhd_vcmax = 149250;
double dhd_jmax = 152040;

double Nleaf = 1 / CNr / SLA_shade;
double Vcmax25 = Nleaf * Flnr * Frn * ACT25;
double Jmax25 = 1.97 * Vcmax25;
double ft_vcmax = exp(dha_vcmax * (1 - 298.15 / (Tleaf + 273.15)) / (298.15 * Rgas));
double fht_vcmax = (1 + exp((298.15 * ds_vcmax - dhd_vcmax) / (298.15 * Rgas))) / (1 + exp((ds_vcmax * (Tleaf + 273.15) - dhd_vcmax) / (Rgas * (Tleaf + 273.15))));
double ft_jmax = exp(dha_jmax * (1 - 298.15 / (Tleaf + 273.15)) / (298.15 * Rgas));
double fht_jmax = (1 + exp((298.15 * ds_jmax - dhd_jmax) / (298.15 * Rgas))) / (1 + exp((ds_jmax * (Tleaf + 273.15) - dhd_jmax) / (Rgas * (Tleaf + 273.15))));
double Vcmax = Vcmax25 * ft_vcmax * fht_vcmax;
double Jmax = Jmax25 * ft_jmax * fht_jmax;

std::cout<<"Vcmax:"<<Vcmax<<"  "<<"Jmax:"<<Jmax<<std::endl;

}