/*The PSN model*/
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "StructHeader.h"

using namespace std;

//#define BYTE float   //方便数据类型的修改
//#define E 2.718281828459

__global__ void SZA_kb_calculation(const double *Lat, const double *Lon, const double *doy, const double *timeSeries,
                                   double *miukb, double *G, double *kb)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //////////////////////不同的时间分辨率，数组结构不同////////////////////////////////
    double pi = 3.141592654;

    double time_zone = ceil((Lon[i] + 7.5) / 15.0) - 1;
    double local = fmod(timeSeries[0] + time_zone, 24.0);

    double LocalTime;
    if (local < 0)
    {
        LocalTime = local + 24;
    }
    else
    {
        LocalTime = local;
    }
    double HourAngle = 15 * (LocalTime - 12);
    double Dec = -23.45 * cos(2 * pi * (doy[0] + 10) / 365);
    double cosSZA = sin(Lat[i] * pi / 180) * sin(Dec * pi / 180) +
                    cos(Lat[i] * pi / 180) * cos(Dec * pi / 180) * cos(HourAngle * pi / 180);
    double SZA = acos(cosSZA) * 180 / pi;
    if (SZA < 0)
    {
        SZA = 0;
    }
    else if (SZA > 88)
    {
        SZA = 88;
    }
    miukb[i] = cos(SZA * pi / 180);
    double Kai = 0.0;
    double Phi1 = 0.5 - 0.633 * Kai - 0.33 * pow(Kai, 2);
    double Phi2 = 0.877 * (1 - 2 * Phi1);
    G[i] = Phi1 + Phi2 * miukb[i];
    kb[i] = G[i] / miukb[i];
}

__global__ void rad_transfer(const double *LAI, const double *kb, const double *G,
                             double *taob, double *betab, double *taod, double *betad)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double coef_scattering_b = 0.1;
    double coef_scattering_d = 0.65;
    double k_p_b = sqrt(1 - coef_scattering_b);
    double k_p_d = sqrt(1 - coef_scattering_d);
    double taob_p = exp(-1 * k_p_b * kb[i] * LAI[i]);
    double kGLAI = k_p_d * G[i] * LAI[i];
    double integral1 = exp(-kGLAI) / 2 * log(1 + 2 / kGLAI);
    double integral2 = exp(-kGLAI) * log(1 + 1 / kGLAI);
    double integral = (integral1 + integral2) / 2;
    double taod_p = (1 - kGLAI) * exp(-kGLAI) + pow(kGLAI, 2) * integral;
    double beta_p_b = (1 - k_p_b) / (1 + k_p_b);
    double beta_p_d = (1 - k_p_d) / (1 + k_p_d);

    // For direct radiationand PAR
    taob[i] = (taob_p * (1 - pow(beta_p_b, 2))) / (1 - pow(beta_p_b, 2) * pow(taob_p, 2));
    betab[i] = (beta_p_b * (1 - pow(taob_p, 2))) / (1 - pow(beta_p_b, 2) * pow(taob_p, 2));

    // For diffuse radiation and PAR
    taod[i] = (taod_p * (1 - pow(beta_p_d, 2))) / (1 - pow(beta_p_d, 2) * pow(taod_p, 2));
    betad[i] = (beta_p_d * (1 - pow(taod_p, 2))) / (1 - pow(beta_p_d, 2) * pow(taod_p, 2));
}

__global__ void rad_partition(const double *Rad, const double *miukb, const double *CorrectionFactor,
                              double *Rad_direct, double *Rad_diffuse)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Partitioning to directand diffuse radiation
    double S0 = 1367 * miukb[i];
    double new_rad;
    if(CorrectionFactor[i]==0)
    {
        new_rad = Rad[i] * CorrectionFactor[i];
    }
    else
    {
        new_rad = Rad[i] / CorrectionFactor[i];
    }
    double AT = new_rad / S0;
    double As = 0.25; // As: fraction of extraterrestrial radiation S0 on overcast days.
    double Bs = 0.5;  // As + Bs: fraction of extraterrestrial radiation S0 on clear days
    if (AT > 0.75)
    {
        AT = 0.75;
    }
    else if (AT < 0.25)
    {
        AT = 0.25;
    }
    double Cf = 1 - (double)((AT - As) / Bs); // Cf: cloudiness fraction, 0 for clear sky and 1 for completely cloudy sky
    double ATc = max(AT, As + Bs);
    double lam = (double)(6.0 / 7.0);  // lam: when sky is clear that a fraction lam of AT is direct.
    double ATb = lam * ATc * (1 - Cf); // ATb: direct radiation fraction.
    if (new_rad == 0)
    {
        Rad_direct[i] = 0;
        Rad_diffuse[i] = 0;
    }
    else
    {
        Rad_direct[i] = (double)(ATb / AT) * new_rad;
        Rad_diffuse[i] = new_rad - Rad_direct[i];
    }
}

__global__ void twoleaf(const double *LAI, const double *Rad_direct, const double *Rad_diffuse,
                        const double *taob, const double *betab, const double *taod, const double *betad, const double *kb, const double *clumping,
                        double *LAI_sun, double *LAI_shade, double *PPFD_sun, double *PPFD_shade)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double kb_diffuse = -log(taod[i]) / LAI[i]; // log(x) returns the natural logarithm of x(to base e).

    // Calculate the ratio of PAR to radiation for direct and diffuse seperately. ES lectures 3-2 P23: PAR = 0.42Sb + 0.6Sd
    double coef_Rad2PAR_b = 0.42;
    double coef_Rad2PAR_d = 0.6;

    // For the absorbed diffuse radiation
    double ARad_diffuse = Rad_diffuse[i] * (1 - taod[i] - betad[i]); // To the total canopy LAI

    // For the absorbed scattered part of direct radiation :
    double Rad_direct_scattered = Rad_direct[i] * taob[i] / (exp(-kb_diffuse * LAI[i]));
    double ARad_direct_scattered = Rad_direct_scattered - Rad_direct[i] * (taob[i] + betab[i]);
    double ARad_direct_direct = Rad_direct[i] * kb[i];

    // Calculate LAI for sunand shade leaf, TWO - LEAF model
    // LAI_sun[i] = ((1 / kb[i]) * (1 - exp(-kb[i] * clumping[i] * LAI[i]))); // Leaf area index of sun leaf(Projected) #LAI is projected LAI.
    LAI_sun[i] = ((1 / kb[i]) * (1 - exp(-kb[i] * LAI[i])));
    LAI_shade[i] = LAI[i] - LAI_sun[i]; // Leaf area index of shade leaf(Projected)

    // Absorbed PPFD by sunlitand shade leaves, which is used as inputs in Farquhar's PSN model
    // Covert W.m - 2 to umol.m - 2.s - 1, umol.m - 2.s - 1
    double APAR_direct_direct = ARad_direct_direct * coef_Rad2PAR_b * 4.57;
    double APAR_direct_scattered = ARad_direct_scattered * coef_Rad2PAR_d * 4.57;
    double APAR_diffuse = ARad_diffuse * coef_Rad2PAR_d * 4.57;

    // PPFD_sun[i] = (APAR_diffuse / LAI[i] * LAI_sun[i] + APAR_direct_scattered / LAI[i] * LAI_sun[i] + APAR_direct_direct) / LAI_sun[i];
    // PPFD_shade[i] = (APAR_diffuse / LAI[i] * LAI_shade[i] + APAR_direct_scattered / LAI[i] * LAI_shade[i]) / LAI_shade[i];
    PPFD_sun[i] = (APAR_diffuse + APAR_direct_scattered) / LAI[i] + APAR_direct_direct / LAI_sun[i];
    PPFD_shade[i] = (APAR_diffuse + APAR_direct_scattered) / LAI[i];
    if (PPFD_shade[i] < 0)
    {
        PPFD_shade[i] = 0;
    }
}

//////////////////////leaf////////////////////
/*
NAME                                  value  samlpe           CNr      Flnr       SLA_total
Water Bodies                            0      WB               0       0          0
Evergreen Needleleaf Forests            1      ENF      use     42      0.04       0.012
Evergreen Broadleaf Forests             2      EBF      use     35      0.046      0.012
Deciduous Needleleaf Forests            3      DNF      use     25      0.055      0.024
Deciduous Broadleaf Forests             4      DBF      use     24      0.08       0.03
Mixed Forests                           5      MF       use     42      0.05       0.012
Closed Shrublands                       6      CShrub   use     42      0.04       0.012
Open Shrublands                         7      OShrub   use     42      0.04       0.012
Woody Savannas                          8      WSavanna use     35      0.04       0.012
Savannas                                9      Savanna  use     35      0.04       0.012
Grasslands                             10      Grass    use     24      0.12       0.045
Permanent Wetlands                     11      PW              
Croplands                              12      Crop     use     25      0.41       0.07
Urban and Built-up Lands               13
Cropland/Natural Vegetation Mosaics    14
Non-Vegetated Lands                    15
*/

__global__ void GetSLA(const double *LAI_sun, const double *LAI_shade,const double *LAI, const double *landcover,
                          double *SLA_sun, double *SLA_shade)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double SLA_ratio = 2 ; //ratio of shaded SLA : sunlit SLA  #BGC
    // double SLA_total = 0.012; //  #m2/gC
    double SLA_total[13] = {0,0.012,0.012,0.024,0.03,0.012,0.012,0.012,0.012,0.012,0.045,0,0.07};
    int index = (int)landcover[i];
    SLA_sun[i] = (LAI_sun[i] + LAI_shade[i] / SLA_ratio) / (LAI[i] / SLA_total[index]);
    SLA_shade[i] = SLA_sun[i] * SLA_ratio;
}

__global__ void GetVcmaxJmax(const double *SLA, const double *Tleaf,const double *landcover,
                            double *Vcmax, double *Jmax, double *Vcmax25)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // double CNr = 42; //gC/gN. C:N of leaves
    // double Flnr = 0.04; //gN in Rubisco / gN. Fraction of leaf N in Rubisco
    double CNr[13] = {0,42,35,25,24,42,42,42,35,35,24,0,25};
    double Flnr[13] = {0,0.04,0.046,0.055,0.08,0.05,0.04,0.04,0.04,0.04,0.12,0,0.41};

    int index = (int)landcover[i];

    double Rgas = 8.314; //ideal gas constant(J.K-1.mol-1)
    double Frn = 7.16; //gRubisco/gN in Rubisco.
    double ACT25 = 60; //umolCO2/gRubisco/s
    // double Q10_ACT = 2.4;
    double dha_vcmax =65330;
    double dha_jmax =43540;
    double ds_vcmax = 485;
    double ds_jmax = 495;
    double dhd_vcmax = 149250;
    double dhd_jmax = 152040;

    double Nleaf = 1 / CNr[index] / SLA[i];
    Vcmax25[i] = Nleaf * Flnr[index] * Frn * ACT25;
    double Jmax25 = 1.97 * Vcmax25[i];
    double ft_vcmax = exp(dha_vcmax * (1 - 298.15 / (Tleaf[i] + 273.15)) / (298.15 * Rgas));
    double fht_vcmax = (1 + exp((298.15 * ds_vcmax - dhd_vcmax) / (298.15 * Rgas))) / (1 + exp((ds_vcmax * (Tleaf[i] + 273.15) - dhd_vcmax) / (Rgas * (Tleaf[i] + 273.15))));
    double ft_jmax = exp(dha_jmax * (1 - 298.15 / (Tleaf[i] + 273.15)) / (298.15 * Rgas));
    double fht_jmax = (1 + exp((298.15 * ds_jmax - dhd_jmax) / (298.15 * Rgas))) / (1 + exp((ds_jmax * (Tleaf[i] + 273.15) - dhd_jmax) / (Rgas * (Tleaf[i] + 273.15))));
    Vcmax[i] = Vcmax25[i] * ft_vcmax * fht_vcmax;
    Jmax[i] = Jmax25 * ft_jmax * fht_jmax;
}

__global__ void PSN_C3(const double *VPD, const double *PPFD, const double *Vcmax,
                       const double *Jmax, const double *Tleaf, const double *RH, const char *gsmodel,
                       double *A, double *GS, double *Rd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double Ca = 400.0; // umol.mol - 1
    double Oi = 210.0; // partial pressure of O2(mmol / mol)
    // double Patm = 101.0;// KPa
    double g1 = 7.5;  // Dimensionless. (Slope)  #Collatz et al.(1991) / Dai et al.(2004) : g1 = 9, g0 = 0.01mol.m - 2.s - 1; Xu's dissertation (Ch7) and Panek and Goldstein (2000): g1=7.5,g0=0.01mol.m-2.s-1
    double g0 = 0.01; // mol.m - 2.s - 1
    double D0 = 2.0;  // KPa ,Xu's dissertation (Ch7) and Panek and Goldstein (2000): D0=2000Pa
    // In BGC4.2 model, alpha = 0.85 / 2.6 for C3 and alpha = 0.85 / 3.5 for C4
    double alpha = 0.85 / 2.6; // quantum yield of electron transport(mol electrons mol - 1 photon)
    // In BGC4.2 model, theta = 0.7
    double theta = 0.7;   // curvature of the light response curve(dimensionless)
    double Rd0 = 0.92;    // umol.m - 2.s - 1(per leaf area) (dark respiration at 25 degree)
    double Q10 = 2.0;     // Dimensionless
    double TrefR = 25.0;  // DegreeC
    double Q10_Kc = 2.1;  // Dimensionless
    double Q10_Ko = 1.2;  // Dimensionless
    double GCtoGW = 1.57; // conversion factor from  GC(Stamatal conductance for CO2) to GW(Stamatal conductance for H2O)
    double Rgas = 8.314;  // J mol-1 K-1
    // Temperature - dependence of Kc, Koand GammaStar, parameters from Medlyn et al. 2001.
    // (Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data)
    // GammaStar is the CO2 compensation point in the absence of photorespiration(umol.mol - 1), the unit of Tleaf should be degree C. (Eq.12 in Medlyn et al. 2001)
    double GammaStar = 42.75 * exp(37830 * ((Tleaf[i] + 273.15) - 298) / (298 * Rgas * (Tleaf[i] + 273.15)));

    // Rd is the rate of dark respiration(maintenance respiration)
    Rd[i] = Rd0 * pow(Q10, (Tleaf[i] - TrefR) / 10.0);

    // Kc is the michaelis constant for CO2(umol.mol - 1)
    // Ko is the michaelis constant for O2(mmol.mol - 1)
    double Kc;
    if (Tleaf[i] > 15)
    {
        Kc = 404.0 * pow(Q10_Kc, ((Tleaf[i] - TrefR) / 10.0));
    }
    else
    {
        Kc = 404.0 * pow(1.8 * Q10_Kc, ((Tleaf[i] - 15) / 10.0)) / Q10_Kc;
    }

    double Ko = 248.0 * pow(Q10_Ko, ((Tleaf[i] - 25) / 10.0));
    double Km = Kc * (1.0 + Oi / Ko);

    // Get the solution of J from the equation : theta * J ^ 2 - (alpha * Q + Jmax) * J + alpha * Q * Jmax = 0 (Medlyn et al. 2001)
    //  J is the rate of electron transport, Q / PPFD is the incident photosynthetically active photon flux density.
    double J = (alpha * PPFD[i] + Jmax[i] - sqrt(pow(alpha * PPFD[i] + Jmax[i], 2) - 4 * alpha * theta * PPFD[i] * Jmax[i])) / (2 * theta);
    double VJ = J / 4.0;

    // Different stomatal conductance model
    //"Ball Berry Leuning" model,gsmodel == "BBL"
    double GSDIVA;
    GSDIVA = g1 / (Ca * (1 + VPD[i] / D0)) / GCtoGW;

    //"Ball Berry" model,gsmodel == "BB"
    // GSDIVA = g1 * RH[i] / 100 / Ca;
    // GSDIVA = GSDIVA / GCtoGW;

    // Get Ac and Aj by solving the quadratic equations, substitution for Ci from the equations below:
    //
    //        Vcmax (Ci - GammaStar)
    //  Ac =  -----------------------  -  Rd         Equation(a)
    //        Ci + Kc (1 + Oi/Ko)
    //
    //          J (Ci - GammaStar)
    //  Aj  =  ----------------------  -   Rd        Equation(b)
    //          4Ci + 8GammaStar
    //
    //  A  =  GS(Ca-Ci)                            Equation(c)
    //
    //  GS = g0 + GSDIVA * A                       Equation(d)

    double CIC;
    double CIJ;
    // double a2;
    if (PPFD[i] == 0)
    {
        CIJ = Ca;
        CIC = Ca;
    }
    else
    {
        // Get CiC from Equation(a)& Equation(c)& Equation(d)
        double a1 = g0 + GSDIVA * (Vcmax[i] - Rd[i]);
        double b1 = (1 - Ca * GSDIVA) * (Vcmax[i] - Rd[i]) + g0 * (Km - Ca) - GSDIVA * (Vcmax[i] * GammaStar + Km * Rd[i]);
        double c1 = -(1 - Ca * GSDIVA) * (Vcmax[i] * GammaStar + Km * Rd[i]) - g0 * Km * Ca;
        CIC = (-b1 + sqrt(b1 * b1 - 4 * a1 * c1)) / (2 * a1);

        // Get CiJ from Equation(b)& Equation(c)& Equation(d)
        double a2 = g0 + GSDIVA * (VJ - Rd[i]);
        double b2 = (1 - Ca * GSDIVA) * (VJ - Rd[i]) + g0 * (2 * GammaStar - Ca) - GSDIVA * (VJ * GammaStar + 2 * GammaStar * Rd[i]);
        double c2 = -(1 - Ca * GSDIVA) * GammaStar * (VJ + 2 * Rd[i]) - g0 * 2 * GammaStar * Ca;
        // double a2 = 4 * g0 + GSDIVA * (J - 4 * Rd[i]);
        // double b2 = (1 - Ca * GSDIVA) * (J - 4 * Rd[i]) + 4 * g0 * (2 * GammaStar - Ca) - GSDIVA * GammaStar * (J + 8 * Rd[i]);
        // double c2 = -(1 - Ca * GSDIVA) * GSDIVA * GammaStar * (J + 8 * Rd[i]) - g0 * 8 * GammaStar * Ca;

        if (a2 == 0)
        {
            CIJ = -c2 / b2;
        }
        else
        {
            CIJ = (-b2 + sqrt(pow(b2, 2) - 4 * a2 * c2)) / (2 * a2);
        }
    }
    double Ac = Vcmax[i] * (CIC - GammaStar) / (CIC + Km);      // the RuBP carboxylase (Rubisco) is limited
    double Aj = VJ * (CIJ - GammaStar) / (CIJ + 2 * GammaStar); // the rate of electron transport is limited
    // double Aj = J * (CIJ - GammaStar) / (4 * CIJ + 8 * GammaStar);
    // double Ap = Vcmax[i] * 0.5;
    // Using min(Ac, Aj) to calculate stomatal conductance(GS)
    A[i] = min(Ac, Aj) + Rd[i];
    // A[i] = min(Ac, Aj);
    GS[i] = g0 + GSDIVA * (A[i] - Rd[i]);

    if (Tleaf[i] < 5)
    {
        A[i] = 0;
    }

    // if(A[i]==Ac)
    // {
    //     A[i]=1;
    // }
    // else if (A[i]==Aj)
    // {
    //     A[i]=0;
    // }
}

__global__ void PSN_C4(const double *VPD, const double *PPFD, const double *Vcmax25,
                       const double *Tleaf, const double *RH, const char *gsmodel,
                       double *A, double *GS, double *Rd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double Ca = 400.0; // umol.mol - 1
    // double Oi = 210.0;   // partial pressure of O2(mmol / mol)
    double Patm = 101.0; // KPa
    double g1 = 7.5;     // Dimensionless. (Slope)  #Collatz et al.(1991) / Dai et al.(2004) : g1 = 9, g0 = 0.01mol.m - 2.s - 1; Xu's dissertation (Ch7) and Panek and Goldstein (2000): g1=7.5,g0=0.01mol.m-2.s-1
    double g0 = 0.01;    // mol.m - 2.s - 1
    double D0 = 2.0;     // KPa ,Xu's dissertation (Ch7) and Panek and Goldstein (2000): D0=2000Pa
    // In BGC4.2 model, alpha = 0.85 / 2.6 for C3 and alpha = 0.85 / 3.5 for C4
    // double alpha = 0.85 / 2.6; // quantum yield of electron transport(mol electrons mol - 1 photon)
    // In BGC4.2 model, theta = 0.7
    // double theta = 0.7;   // curvature of the light response curve(dimensionless)
    double Rd0 = 0.92;   // umol.m - 2.s - 1(per leaf area) (dark respiration at 25 degree)
    double Q10 = 2.0;    // Dimensionless
    double TrefR = 25.0; // DegreeC
    // double Q10_Kc = 2.1;  // Dimensionless
    // double Q10_Ko = 1.2;  // Dimensionless
    double GCtoGW = 1.57; // conversion factor from  GC(Stamatal conductance for CO2) to GW(Stamatal conductance for H2O)
    // double Rgas = 8.314;

    /////////////////////C4///////////////////////////
    double C4_alphs = 0.04;
    double C4_beta = 0.93;
    double C4_theta = 0.83;
    // double k = 0.7; // mol/m2/s/bar
    double m = 3.0;
    double P = 100000;
    double Ps = 40000000;
    double b = 0.08;

    // Rd is the rate of dark respiration(maintenance respiration)
    Rd[i] = Rd0 * pow(Q10, (Tleaf[i] - TrefR) / 10.0);

    // Different stomatal conductance model
    //"Ball Berry Leuning" model,gsmodel == "BB"
    double GSDIVA;
    GSDIVA = g1 / (Ca * (1 + VPD[i] / D0)) / GCtoGW;

    //"Ball Berry" model,gsmodel == "BB"
    // GSDIVA = g1 * RH[i] / 100 / Ca;
    // GSDIVA = GSDIVA / GCtoGW;

    double Vt = Vcmax25[i] * pow(Q10, (Tleaf[i] - 25) / 10.0) / (1 + exp((Tleaf[i] - 40) * 0.3))/ (1 + exp((15 - Tleaf[i]) * 0.2)) ;
    double Rt = Rd0 * pow(Q10, (Tleaf[i] - 25) / 10.0) / (1 + exp((Tleaf[i] - 55) * 1.3));
    double k = Vcmax25[i] * 20000 / pow(10,6);
    double Kt = k * pow(Q10, (Tleaf[i] - 25) / 10.0);
    double M1 = (Vt + C4_alphs * PPFD[i] - sqrt(pow(Vt + C4_alphs * PPFD[i], 2) - 4 * C4_theta * Vt * C4_alphs * PPFD[i])) / (C4_theta * 2);
    double a_prime = C4_beta * m * RH[i] * P / Ps;
    double b_prime = C4_beta * b - (C4_beta * m * RH[i] * Rt * P / Ps) + Kt * 1.6 - M1 * m * RH[i] * P / Ps - Kt * m * RH[i];
    double c_prime = Rt * M1 * m * RH[i] * P / Ps - M1 * b + Rt * Kt * m * RH[i] - Kt * b * Ps / P - Kt * Rt * 1.6 + M1 * Kt * m * RH[i] - Kt * M1 * 1.6;
    double d_prime = M1 * Kt * b * Ps / P + M1 * Kt * Rt * 1.6 - M1 * Kt * m * RH[i] * Rt;
    double Q = (pow(b_prime / a_prime, 2) - c_prime / a_prime * 3) / 9.0;
    double R = (pow(b_prime / a_prime, 3) * 2.0 - b_prime * c_prime * 9.0 / pow(a_prime, 2) + d_prime / a_prime * 27.0) / 54.0;
    double aa = R / pow(Q, 1.5);
    double S;
    if (aa > 1)
    {
        S = acos(0.0);
    }
    else if (aa < -1)
    {
        S = acos(0.0);
    }
    else
    {
        S = acos(aa);
    }

    /////////////Ap//////////////////////
    double a_ap = GSDIVA * k * pow(10, 6) / (Patm * 1000);
    double b_ap = g0 + (k * pow(10, 6) / (Patm * 1000)) - (GSDIVA * Ca * k * pow(10, 6) / (Patm * 1000));
    double c_ap = -g0 * Ca;
    double CiP = (sqrt(pow(b_ap, 2) - 4 * a_ap * c_ap) - b_ap) / (a_ap * 2);
    double Ap = k * pow(10, 6) * CiP / (Patm * 1000);
    // Using min(A,Aj) to calculate stomatal conductance(GS)
    A[i] = pow(Q, 0.5) * (-2) * cos((S + 12.56) / 3.0) - b_prime / (a_prime * 3.0);
    A[i] = min(A[i], Ap) + Rd[i];

    GS[i] = g0 + GSDIVA * (A[i] - Rd[i]);

    if (Tleaf[i] < 5)
    {
        A[i] = 0;
    }

    // A[i] = k;
}

__global__ void inputData(const double *Temp,const double *SH, const double *Pressure, 
                          double *VPD, double *RH)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double es = 0.61078 * exp((17.27 * Temp[i]) / (Temp[i] + 237.3));
    double ea = 1.6077 * SH[i] * Pressure[i] * 0.001;
    VPD[i] = es - ea;
    if (VPD[i] < 0)
    {
        VPD[i] = 0;
    }
    RH[i] = (ea / es) * 100;
    if (RH[i] < 0)
    {
        RH[i] = 0;
    }
    else if (RH[i] > 100)
    {
        RH[i] = 100;
    }
}

__global__ void inputLai(const double *p0, const double *p1, const double *p2,
                         const double *p3, const double *p4, const double *p5, const double *t, double *LaiData)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    LaiData[i] = (p0[i] + (double)p1[i] / (1 + exp(-p2[i] * (t[0] - p3[i]))) - (double)p1[i] / (1 + exp(-p4[i] * (t[0] - p5[i])))) * 0.1;

    if (p0[i] == 0 && p1[i] == 0 && p2[i] == 0 && p3[i] == 0 && p4[i] == 0 && p5[i] == 0)
    {
        LaiData[i] = NAN;
    }

    if (LaiData[i] < 0)
    {
        LaiData[i] = 0;
    }
}

__global__ void total_PSN(const double *A_sun, const double *GS_sun, const double *Rd_sun,
                          const double *A_shade, const double *GS_shade, const double *Rd_shade,
                          const double *LAI_sun, const double *LAI_shade,
                          double *total_A, double *total_GS, double *total_Rd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    total_A[i] = A_sun[i] * LAI_sun[i] + A_shade[i] * LAI_shade[i];
    total_GS[i] = GS_sun[i] * LAI_sun[i] + GS_shade[i] * LAI_shade[i];
    total_Rd[i] = Rd_sun[i] * LAI_sun[i] + Rd_shade[i] * LAI_shade[i];
}

__global__ void Photosynthesis(const double *A_C3, const double *GS_C3, const double *Rd_C3,
                               const double *A_C4, const double *GS_C4, const double *Rd_C4, const double *PercentC4,
                               double *Photosynthesis_A, double *Photosynthesis_GS, double *Photosynthesis_Rd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    Photosynthesis_A[i] = A_C3[i] * (1 - PercentC4[i]) + A_C4[i] * PercentC4[i];
    Photosynthesis_GS[i] = GS_C3[i] * (1 - PercentC4[i]) + GS_C4[i] * PercentC4[i];
    Photosynthesis_Rd[i] = Rd_C3[i] * (1 - PercentC4[i]) + Rd_C4[i] * PercentC4[i];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void inputLai_C(int blocksPerGrid, int threadsPerBlock, const double *p0, const double *p1, const double *p2,
                           const double *p3, const double *p4, const double *p5, const double *t, double *LaiData)
{
    inputLai<<<blocksPerGrid, threadsPerBlock>>>(p0, p1, p2, p3, p4, p5, t, LaiData);
}

extern "C" void inputData_C(int blocksPerGrid, int threadsPerBlock, const double *Temp,const double *SH, const double *Pressure, 
                          double *VPD, double *RH)
{
    inputData<<<blocksPerGrid, threadsPerBlock>>>(Temp, SH, Pressure,VPD, RH);
}

extern "C" void SZA_kb_calculation_C(int blocksPerGrid, int threadsPerBlock, const double *Lat, const double *Lon, const double *doy, const double *timeSeries,
                                     double *miukb, double *G, double *kb)
{
    SZA_kb_calculation<<<blocksPerGrid, threadsPerBlock>>>(Lat, Lon, doy, timeSeries, miukb, G, kb);
}

extern "C" void rad_transfer_C(int blocksPerGrid, int threadsPerBlock, const double *LAI, const double *kb, const double *G,
                               double *taob, double *betab, double *taod, double *betad)
{
    rad_transfer<<<blocksPerGrid, threadsPerBlock>>>(LAI, kb, G, taob, betab, taod, betad);
}

extern "C" void rad_partition_C(int blocksPerGrid, int threadsPerBlock, const double *Rad, const double *miukb,const double *CorrectionFactor,
                                double *Rad_direct, double *Rad_diffuse)
{
    rad_partition<<<blocksPerGrid, threadsPerBlock>>>(Rad, miukb,CorrectionFactor, Rad_direct, Rad_diffuse);
}

extern "C" void twoleaf_C(int blocksPerGrid, int threadsPerBlock, const double *LAI, const double *Rad_direct, const double *Rad_diffuse,
                          const double *taob, const double *betab, const double *taod, const double *betad, const double *kb, const double *clumping,
                          double *LAI_sun, double *LAI_shade, double *PPFD_sun, double *PPFD_shade)
{
    twoleaf<<<blocksPerGrid, threadsPerBlock>>>(LAI, Rad_direct, Rad_diffuse, taob, betab, taod, betad, kb, clumping,
                                                LAI_sun, LAI_shade, PPFD_sun, PPFD_shade);
}


extern "C" void GetSLA_C(int blocksPerGrid, int threadsPerBlock, const double *LAI_sun, const double *LAI_shade,const double *LAI, const double *landcover,
                          double *SLA_sun, double *SLA_shade)
{
    GetSLA<<<blocksPerGrid, threadsPerBlock>>>(LAI_sun, LAI_shade, LAI, landcover,SLA_sun, SLA_shade);
}

extern "C" void GetVcmaxJmax_C(int blocksPerGrid, int threadsPerBlock, const double *SLA, const double *Tleaf,const double *landcover,double *Vcmax, double *Jmax, double *Vcmax25)
{
    GetVcmaxJmax<<<blocksPerGrid, threadsPerBlock>>>(SLA, Tleaf,landcover,Vcmax, Jmax,Vcmax25);
}


extern "C" void PSN_C3_C(int blocksPerGrid, int threadsPerBlock, const double *VPD, const double *PPFD, const double *Vcmax,
                         const double *Jmax, const double *Tleaf, const double *RH, const char *gsmodel,
                         double *A, double *GS, double *Rd)
{
    PSN_C3<<<blocksPerGrid, threadsPerBlock>>>(VPD, PPFD, Vcmax, Jmax, Tleaf, RH, gsmodel, A, GS, Rd);
}

extern "C" void PSN_C4_C(int blocksPerGrid, int threadsPerBlock, const double *VPD, const double *PPFD, const double *Vcmax25,
                       const double *Tleaf, const double *RH, const char *gsmodel,
                       double *A, double *GS, double *Rd)
{
    PSN_C4<<<blocksPerGrid, threadsPerBlock>>>(VPD, PPFD, Vcmax25, Tleaf, RH, gsmodel, A, GS, Rd);
}

extern "C" void total_PSN_C3(int blocksPerGrid, int threadsPerBlock, const double *A_sun, const double *GS_sun, const double *Rd_sun,
                             const double *A_shade, const double *GS_shade, const double *Rd_shade,
                             const double *LAI_sun, const double *LAI_shade,
                             double *total_A, double *total_GS, double *total_Rd)
{
    total_PSN<<<blocksPerGrid, threadsPerBlock>>>(A_sun, GS_sun, Rd_sun,
                                                  A_shade, GS_shade, Rd_shade, LAI_sun, LAI_shade,
                                                  total_A, total_GS, total_Rd);
}

extern "C" void total_PSN_C4(int blocksPerGrid, int threadsPerBlock, const double *A_sun, const double *GS_sun, const double *Rd_sun,
                             const double *A_shade, const double *GS_shade, const double *Rd_shade,
                             const double *LAI_sun, const double *LAI_shade,
                             double *total_A, double *total_GS, double *total_Rd)
{
    total_PSN<<<blocksPerGrid, threadsPerBlock>>>(A_sun, GS_sun, Rd_sun,
                                                  A_shade, GS_shade, Rd_shade, LAI_sun, LAI_shade,
                                                  total_A, total_GS, total_Rd);
}

extern "C" void Photosynthesis_C(int blocksPerGrid, int threadsPerBlock, const double *A_C3, const double *GS_C3, const double *Rd_C3,
                                 const double *A_C4, const double *GS_C4, const double *Rd_C4, const double *PercentC4,
                                 double *Photosynthesis_A, double *Photosynthesis_GS, double *Photosynthesis_Rd)
{
    Photosynthesis<<<blocksPerGrid, threadsPerBlock>>>(A_C3, GS_C3, Rd_C3,
                                                       A_C4, GS_C4, Rd_C4, PercentC4,
                                                       Photosynthesis_A, Photosynthesis_GS, Photosynthesis_Rd);
}