/*The PSN model*/
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "StructHeader.h"
// #include "PSN.cuh"
using namespace std;

//#define BYTE float   //方便数据类型的修改
//#define E 2.718281828459


__global__ void SZA_kb_calculation(const double* Lat, const double* Lon, const double* doy, const double* timeSeries,
    double* miukb, double* G, double* kb)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //////////////////////不同的时间分辨率，数组结构不同////////////////////////////////
    double pi = 3.141592654;

    double time_zone = ceil((Lon[i] + 7.5) / 15.0) - 1;
    float local1;
    
    modff((timeSeries[0] + time_zone)/24.0, &local1);
    double local = (double)local1;
    
    double LocalTime = 0;
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

__global__ void rad_transfer(const double* LAI, const double* kb, const double* G,
    double* taob, double* betab, double* taod, double* betad)
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

__global__ void rad_partition(const double* Rad, const double* miukb,
    double* Rad_direct, double* Rad_diffuse)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //Partitioning to directand diffuse radiation
    double S0 = 1367 * miukb[i];
    double AT = Rad[i] / S0;
    double As = 0.25;  // As: fraction of extraterrestrial radiation S0 on overcast days.
    double Bs = 0.5; //As + Bs: fraction of extraterrestrial radiation S0 on clear days
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
    double lam = (double)(6.0 / 7.0); //lam: when sky is clear that a fraction lam of AT is direct.
    double ATb = lam * ATc * (1 - Cf);// ATb: direct radiation fraction.
    if (Rad[i] == 0)
    {
        Rad_direct[i] = 0;
        Rad_diffuse[i] = 0;
    }
    else
    {
        Rad_direct[i] = (double)(ATb / AT) * Rad[i];
        Rad_diffuse[i] = Rad[i] - Rad_direct[i];
    }
}

__global__ void twoleaf(const double* LAI, const double* Rad_direct, const double* Rad_diffuse,
    const double* taob, const double* betab, const double* taod, const double* betad, const double* kb,
    double* LAI_sun, double* LAI_shade, double* PPFD_sun, double* PPFD_shade)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double kb_diffuse = -log(taod[i]) / LAI[i];// log(x) returns the natural logarithm of x(to base e).

    //Calculate the ratio of PAR to radiation for direct and diffuse seperately. ES lectures 3-2 P23: PAR = 0.42Sb + 0.6Sd
    double coef_Rad2PAR_b = 0.42;
    double coef_Rad2PAR_d = 0.6;

    //For the absorbed diffuse radiation
    double ARad_diffuse = Rad_diffuse[i] * (1 - taod[i] - betad[i]);//To the total canopy LAI

    // For the absorbed scattered part of direct radiation :
    double Rad_direct_scattered = Rad_direct[i] * taob[i] / (exp(-kb_diffuse * LAI[i]));
    double ARad_direct_scattered = Rad_direct_scattered - Rad_direct[i] * (taob[i] + betab[i]);
    double ARad_direct_direct = Rad_direct[i] * kb[i];

    // Calculate LAI for sunand shade leaf, TWO - LEAF model
    LAI_sun[i] = ((1 / kb[i]) * (1 - exp(-kb[i] * LAI[i])));  // Leaf area index of sun leaf(Projected) #LAI is projected LAI.
    LAI_shade[i] = LAI[i] - LAI_sun[i];  // Leaf area index of shade leaf(Projected)

    //Absorbed PPFD by sunlitand shade leaves, which is used as inputs in Farquhar's PSN model
    //Covert W.m - 2 to umol.m - 2.s - 1, umol.m - 2.s - 1
    double APAR_direct_direct = ARad_direct_direct * coef_Rad2PAR_b * 4.57;
    double APAR_direct_scattered = ARad_direct_scattered * coef_Rad2PAR_d * 4.57;
    double APAR_diffuse = ARad_diffuse * coef_Rad2PAR_d * 4.57;

    PPFD_sun[i] = (APAR_diffuse / LAI[i] * LAI_sun[i] + APAR_direct_scattered / LAI[i] * LAI_sun[i] + APAR_direct_direct) / LAI_sun[i];
    PPFD_shade[i] = (APAR_diffuse / LAI[i] * LAI_shade[i] + APAR_direct_scattered / LAI[i] * LAI_shade[i]) / LAI_shade[i];
}

__global__ void PSN_C4(const double* VPD, const double* PPFD, const double* Vcmax,
    const double* Jmax, const double* Tleaf, const double* RH, const char* gsmodel,
    double* A, double* GS, double* Rd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double Ca = 400.0;// umol.mol - 1
    double Oi = 210.0;// partial pressure of O2(mmol / mol)
    double Patm = 101.0;// KPa
    double g1 = 7.5;// Dimensionless. (Slope)  #Collatz et al.(1991) / Dai et al.(2004) : g1 = 9, g0 = 0.01mol.m - 2.s - 1; Xu's dissertation (Ch7) and Panek and Goldstein (2000): g1=7.5,g0=0.01mol.m-2.s-1
    double g0 = 0.01;// mol.m - 2.s - 1
    double D0 = 2.0;// KPa ,Xu's dissertation (Ch7) and Panek and Goldstein (2000): D0=2000Pa
    // In BGC4.2 model, alpha = 0.85 / 2.6 for C3 and alpha = 0.85 / 3.5 for C4
    double alpha = 0.85 / 2.6;// quantum yield of electron transport(mol electrons mol - 1 photon)
    // In BGC4.2 model, theta = 0.7
    double theta = 0.7;// curvature of the light response curve(dimensionless)
    double Rd0 = 0.92;// umol.m - 2.s - 1(per leaf area) (dark respiration at 25 degree)
    double Q10 = 2.0;// Dimensionless
    double TrefR = 25.0;// DegreeC
    double Q10_Kc = 2.1;// Dimensionless
    double Q10_Ko = 1.2;// Dimensionless
    double GCtoGW = 1.57;// conversion factor from  GC(Stamatal conductance for CO2) to GW(Stamatal conductance for H2O)
    double Rgas = 8.314;

    /////////////////////C4///////////////////////////
    double C4_alphs = 0.04;
    double C4_beta = 0.93;
    double C4_theta = 0.83;
    double k = 0.7; //mol/m2/s/bar
    double m = 3.0;
    double P = 100000;
    double Ps = 40000000;
    double b = 0.08;

    // Temperature - dependence of Kc, Koand GammaStar, parameters from Medlyn et al. 2001.
    // (Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data)
    // GammaStar is the CO2 compensation point in the absence of photorespiration(umol.mol - 1), the unit of Tleaf should be degree C. (Eq.12 in Medlyn et al. 2001)
    double GammaStar = 42.75 * exp(37830 * ((Tleaf[i] + 273.15) - 298) / (298 * Rgas * (Tleaf[i] + 273.15)));

    //Rd is the rate of dark respiration(maintenance respiration)
    Rd[i] = Rd0 * pow(Q10, (Tleaf[i] - TrefR) / 10.0);

    //Kc is the michaelis constant for CO2(umol.mol - 1)
    //Ko is the michaelis constant for O2(mmol.mol - 1)
    double Kc;
    if (Tleaf[i] > 15)
    {
        Kc = 404.0 * pow(Q10_Kc, ((Tleaf[i] - TrefR) / 10.0));
    }
    else
    {
        Kc = 404.0 *1.8* pow( Q10_Kc, ((Tleaf[i] - 15) / 10.0)) / Q10_Kc;
    }

    double Ko = 248.0 * pow(Q10_Ko, ((Tleaf[i] - 25) / 10.0));
    double Km = Kc * (1.0 + Oi / Ko);

    //Get the solution of J from the equation : theta * J ^ 2 - (alpha * Q + Jmax) * J + alpha * Q * Jmax = 0 (Medlyn et al. 2001)
    // J is the rate of electron transport, Q / PPFD is the incident photosynthetically active photon flux density.
    double J = (alpha * PPFD[i] + Jmax[i] - sqrt(pow(alpha * PPFD[i] + Jmax[i], 2) - 4 * alpha * theta * PPFD[i] * Jmax[i])) / (2 * theta);
    double VJ = J / 4.0;

    //Different stomatal conductance model
    //"Ball Berry Leuning" model,gsmodel == "BB"
    double GSDIVA;
    GSDIVA = g1 / (Ca * (1 + VPD[i] / D0))/ GCtoGW;

    //"Ball Berry" model,gsmodel == "BB"
    //GSDIVA = g1 * RH[i] / 100 / Ca;
    //GSDIVA = GSDIVA / GCtoGW;

    double Vt=Vcmax[i]*pow(Q10,(Tleaf[i] - 25) / 10.0)/(1+exp((13-Tleaf[i])*0.3))/(1+exp((Tleaf[i]-36)*0.3));
    double Rt=Rd0*pow(Q10,(Tleaf[i] - 25) / 10.0)/(1+exp((Tleaf[i]-55)*1.3));
    double Kt=k*pow(Q10,(Tleaf[i] - 25) / 10.0);
    double M1=(Vt+C4_alphs*PPFD[i]-sqrt(pow(Vt+C4_alphs*PPFD[i],2)-4*C4_theta*Vt*C4_alphs*PPFD[i]))/(C4_theta*2);
    double a_prime=C4_beta*m*RH[i]*P/Ps;
    double b_prime=C4_beta*b-(C4_beta*m*RH[i]*Rt*P/Ps)+Kt*1.6-M1*m*RH[i]*P/Ps-Kt*m*RH[i];
    double c_prime=Rt*M1*m*RH[i]*P/Ps-M1*b+Rt*Kt*m*RH[i]-Kt*b*Ps/P-Kt*Rt*1.6+M1*Kt*m*RH[i]-Kt*M1*1.6;
    double d_prime=M1*Kt*b*Ps/P+M1*Kt*Rt*1.6-M1*Kt*m*RH[i]*Rt;
    double Q=(pow(b_prime/a_prime,2)-c_prime/a_prime*3)/9.0;
    double R=(pow(b_prime/a_prime,3)*2.0-b_prime*c_prime*9.0/pow(a_prime,2)+d_prime/a_prime*27.0)/54.0;
    double aa=R/pow(Q,1.5);
    double S;
    if (aa > 1)
    {
        S=acos(0.0);
    }
    else if (aa < -1 )
    {
        S=acos(0.0);
    }
    else
    {
        S=acos(aa);
    }

    /////////////Ap//////////////////////
    double a_ap=GSDIVA*k*pow(10,6)/(Patm*1000);
    double b_ap=g0+(k*pow(10,6)/(Patm*1000))-(GSDIVA*Ca*k*pow(10,6)/(Patm*1000));
    double c_ap=-g0*Ca;
    double CiP=(sqrt(pow(b_ap,2)-4*a_ap*c_ap)-b_ap)/(a_ap*2);
    double Ap=k*pow(10,6)*CiP/(Patm*1000);
    //Using min(A,Aj) to calculate stomatal conductance(GS)
    A[i]=pow(Q,0.5)*(-2)*cos((S+12.56)/3.0)-b_prime/(a_prime*3.0);
    A[i]=min(A[i],Ap);
    GS[i]=g0+GSDIVA*(A[i]-Rd[i]);

    if (Tleaf[i] <= 5)
    {
        A[i] = 0;
    }

}

__global__ void inputData(const double* Lon, const double* Lat, const double* Temp,
    const double* Rad, const double* SH, const double* Pressure, const double* LAI,
    double* VPD, double* RH, double* Vcmax, double* Jmax)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double Rgas = 8.314;//deal gas constant(J.K-1.mol-1)
    double Vcmax25 = 62.5; // umol.m - 2.s - 1  (per leaf area)(CLM4.5)
    double Jmax25 = 95.49; // umol.m - 2.s - 1  (per leaf area)
    // Parameters for T - dependence of Vcmax and Jmax(peaked function, 1942)
    double EaV = 65330; //J.mol - 1  (CLM4.5)
    double EaJ = 43540; //J.mol - 1  (CLM4.5)

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

    //Calculate Vcmax based on Vcmax25and leaf temperature
    //Temp=Tleaf
    double T_Vcmax = exp(EaV * ((Temp[i] + 273.15) - 298) / (298 * Rgas * (Temp[i] + 273.15)));
    Vcmax[i] = Vcmax25 * T_Vcmax;
    //Calculate Jmax for Aj based on Jmax25 and leaf temperature
    double T_Jmax = exp(EaJ * ((Temp[i] + 273.15) - 298) / (298 * Rgas * (Temp[i] + 273.15)));
    Jmax[i] = Jmax25 * T_Jmax;

    /* Vcmax_sun = Vcmax
         Vcmax_shade = Vcmax
         Jmax_sun = Jmax
         Jmax_shade = Jmax*/

}

__global__ void inputLai(const double* p0, const double* p1, const double* p2,
    const double* p3, const double* p4, const double* p5, const double* t, double* LaiData)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    LaiData[i] = (p0[i] + (double)p1[i] / (1 + exp(-p2[i] * (t[0] - p3[i]))) - (double)p1[i] / (1 + exp(-p4[i] * (t[0] - p5[i]))))*0.1;
    if(LaiData[i] < 0)
    {
        LaiData[i]=0;
    }
}


__global__ void total_PSN(const double* A_sun, const double* GS_sun, const double* Rd_sun,
    const double* A_shade, const double* GS_shade, const double* Rd_shade,
    const double* LAI_sun, const double* LAI_shade,
    double* total_A, double* total_GS, double* total_Rd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    total_A[i] = A_sun[i] * LAI_sun[i] + A_shade[i] * LAI_shade[i];
    total_GS[i] = GS_sun[i] * LAI_sun[i] + GS_shade[i] * LAI_shade[i];
    total_Rd[i] = Rd_sun[i] * LAI_sun[i] + Rd_shade[i] * LAI_shade[i];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void inputLai_C(int blocksPerGrid, int threadsPerBlock, const double* p0, const double* p1, const double* p2,
    const double* p3, const double* p4, const double* p5, const double* t, double* LaiData)
{
    inputLai << <blocksPerGrid, threadsPerBlock >> > (p0, p1, p2, p3, p4, p5, t, LaiData);
}

extern "C" void inputData_C(int blocksPerGrid, int threadsPerBlock, const double* Lon, const double* Lat, const double* Temp,
    const double* Rad, const double* SH, const double* Pressure, const double* LAI,
    double* VPD, double* RH, double* Vcmax, double* Jmax)
{
    inputData << <blocksPerGrid, threadsPerBlock >> > (Lon, Lat, Temp, Rad, SH, Pressure,
        LAI, VPD, RH, Vcmax, Jmax);
}

extern "C" void SZA_kb_calculation_C(int blocksPerGrid, int threadsPerBlock, const double* Lat, const double* Lon, const double* doy, const double* timeSeries,
    double* miukb, double* G, double* kb)
{
    SZA_kb_calculation << <blocksPerGrid, threadsPerBlock >> > (Lat, Lon, doy, timeSeries, miukb, G, kb);
}

extern "C" void rad_transfer_C(int blocksPerGrid, int threadsPerBlock, const double* LAI, const double* kb, const double* G,
    double* taob, double* betab, double* taod, double* betad)
{
    rad_transfer << <blocksPerGrid, threadsPerBlock >> > (LAI, kb, G, taob, betab, taod, betad);
}

extern "C" void rad_partition_C(int blocksPerGrid, int threadsPerBlock, const double* Rad, const double* miukb,
    double* Rad_direct, double* Rad_diffuse)
{
    rad_partition << <blocksPerGrid, threadsPerBlock >> > (Rad, miukb, Rad_direct, Rad_diffuse);
}

extern "C" void twoleaf_C(int blocksPerGrid, int threadsPerBlock, const double* LAI, const double* Rad_direct, const double* Rad_diffuse,
    const double* taob, const double* betab, const double* taod, const double* betad, const double* kb,
    double* LAI_sun, double* LAI_shade, double* PPFD_sun, double* PPFD_shade)
{
    twoleaf << <blocksPerGrid, threadsPerBlock >> > (LAI, Rad_direct, Rad_diffuse, taob, betab, taod, betad, kb,
        LAI_sun, LAI_shade, PPFD_sun, PPFD_shade);
}

extern "C" void PSN_C4_C(int blocksPerGrid, int threadsPerBlock, const double* VPD, const double* PPFD, const double* Vcmax,
    const double* Jmax, const double* Tleaf, const double* RH, const char* gsmodel,
    double* A, double* GS, double* Rd)
{
    PSN_C4 << <blocksPerGrid, threadsPerBlock >> > (VPD, PPFD, Vcmax, Jmax, Tleaf, RH, gsmodel,A, GS, Rd);
}

extern "C" void total_PSN_C4(int blocksPerGrid, int threadsPerBlock, const double* A_sun, const double* GS_sun, const double* Rd_sun,
    const double* A_shade, const double* GS_shade, const double* Rd_shade,
    const double* LAI_sun, const double* LAI_shade,
    double* total_A, double* total_GS, double* total_Rd)
{
    total_PSN << <blocksPerGrid, threadsPerBlock >> > (A_sun, GS_sun, Rd_sun,
        A_shade, GS_shade, Rd_shade, LAI_sun, LAI_shade,
        total_A, total_GS, total_Rd);
}
