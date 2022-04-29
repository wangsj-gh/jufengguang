#include <fstream>
#include <iomanip>
#include <ctime>

#include "StructHeader.h"
#include "PSN.h"
#include "GetDayData.h"
#include "Respiration.h"
#include "Tif_Read.h"
#include "DataSet_Read.h"
#include "DataSet_Write.h"

#include "Spatiotemporal_resolution.h"
#include "DataManager.h"

using namespace std;

int main(void)
{
    clock_t start = clock();
    GDALAllRegister();

    /////////////////////////Resolution//////////////////////
    int TempResolution = 3;
    int SpaResolution = 500;
    std::cout << "begin SpatiotemporalDeque" << std::endl;

    std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque = new std::deque<SpatiotemporalStruct *>;
    Spatiotemporal *SpatiotemporalInput = new Spatiotemporal();
    SpatiotemporalInput->ReadSpatiotemporal(TempResolution, SpaResolution, SpatiotemporalDeque);

    std::cout << "end SpatiotemporalDeque" << std::endl;

    ////////////////////////////DatasetInput///////////////////////
    std::cout << "begin DatasetInput" << std::endl;

    std::deque<InformStruct *> *OutLaiInputDeque = new std::deque<InformStruct *>;
    std::deque<double *> *ClumpIndexInputDeque = new std::deque<double *>;
    std::deque<double *> *PercentC4InputDeque = new std::deque<double *>;
    std::deque<double *> *LandCoverInputDeque = new std::deque<double *>;

    DataSet_Read *DatasetInput = new DataSet_Read();
    DatasetInput->Init();
    DatasetInput->ReadDataset();
    DatasetInput->VerifyData();
    DatasetInput->TransDataset(OutLaiInputDeque, ClumpIndexInputDeque, PercentC4InputDeque, LandCoverInputDeque);
    DatasetInput->Release();
    std::cout << "end DatasetInput" << std::endl;

    //////////////////////////Netcdf/////////////////
    std::cout << "begin Netcdf" << std::endl;

    std::map<std::string, std::deque<VarInfo *> *> VarInfoMapInputDeque;
    DataManager *p_DataManager = new DataManager();
    p_DataManager->Init();
    p_DataManager->CollectFilesPath();
    p_DataManager->ReadFiles();
    VarInfoMapInputDeque = p_DataManager->GetDataMap();

    std::cout << "end Netcdf" << std::endl;
    clock_t end1 = clock();
    std::cout << "read data time:" << (double)(end1 - start) / CLOCKS_PER_SEC << std::endl;
    //////////////////////////////PSN/////////////////////////////
    std::cout << "begin PSN" << std::endl;

    std::deque<double *> *GppDeque3h = new std::deque<double *>;

    PSN *ModelPsn = new PSN();
    ModelPsn->Init(OutLaiInputDeque);
    ModelPsn->Inneed();
    ModelPsn->PSNGPU(OutLaiInputDeque, SpatiotemporalDeque, ClumpIndexInputDeque,
                     PercentC4InputDeque, VarInfoMapInputDeque, GppDeque3h);
    ModelPsn->Release();

    std::cout << "end PSN" << std::endl;

    clock_t end2 = clock();
    std::cout << "GPU time:" << (double)(end2 - end1) / CLOCKS_PER_SEC << std::endl;
    ////////////////get day data/////////////////////
    // std::cout << "begin GetDayDeque" << std::endl;

    // std::deque<double *> *DayDeque = new std::deque<double *>;

    // GetDayData *ModelDayGpp = new GetDayData();
    // ModelDayGpp->Init(OutLaiInputDeque, SpatiotemporalDeque);
    // ModelDayGpp->Inneed();
    // ModelDayGpp->GetDayDataGPU(GppDeque3h, DayDeque);
    // ModelDayGpp->Release();

    // std::cout << "end DayDeque" << std::endl;

    // const char *file1 = "../dataGdals/Lon_day";
    // DataSet_Write *DatasetC3A3h = new DataSet_Write();
    // DatasetC3A3h->ResultToTif(file1, OutLaiInputDeque, GppDeque3h);
    // clock_t end3 = clock();
    // std::cout << "write time:" << (double)(end3 - end2) / CLOCKS_PER_SEC << std::endl;

    ////////////////////////get day gpp//////////////////////////////

    std::cout << "begin DayGppDeque" << std::endl;

    std::deque<double *> *DayGppDeque = new std::deque<double *>;

    GetDayData *ModelGetDayGpp = new GetDayData();
    ModelGetDayGpp->Init(OutLaiInputDeque, SpatiotemporalDeque);
    ModelGetDayGpp->Inneed();
    ModelGetDayGpp->GetDayDataGPU(GppDeque3h, OutLaiInputDeque, SpatiotemporalDeque, DayGppDeque);
    ModelGetDayGpp->Release();

    std::cout << "end DayGppDeque" << std::endl;

    const char *file2 = "../dataGdals/C3GppDay2007";
    DataSet_Write *DatasetC3A = new DataSet_Write();
    DatasetC3A->ResultToTif(file2, OutLaiInputDeque, DayGppDeque);
    clock_t end3 = clock();
    std::cout << "write time:" << (double)(end3 - end2) / CLOCKS_PER_SEC << std::endl;
    /*
    //////////////get year data/////////////////////
    std::cout << "begin GetYearDeque" << std::endl;

    std::deque<double *> *YearGppDeque = new std::deque<double *>;

    GetDayData *ModelDayGpp = new GetDayData();
    ModelDayGpp->Init(OutLaiInputDeque, SpatiotemporalDeque);
    ModelDayGpp->Inneed();
    ModelDayGpp->GetYearDataGPU(DayGppDeque, YearGppDeque);
    ModelDayGpp->Release();

    std::cout << "end DayDeque" << std::endl;

    const char *file1 = "../dataGdals/C3GppYear";
    DataSet_Write *DatasetC3A3h = new DataSet_Write();
    DatasetC3A3h->ResultToTif(file1, OutLaiInputDeque, YearGppDeque);

    ////////////////////////get day Tleaf//////////////////////////////

    std::cout << "begin DayTleafDeque" << std::endl;

    std::deque<double *> *DayTleafDeque = new std::deque<double *>;

    GetDayData *ModelGetDayTleaf = new GetDayData();
    ModelGetDayTleaf->Init(OutLaiInputDeque, SpatiotemporalDeque);
    ModelGetDayTleaf->Inneed();
    ModelGetDayTleaf->GetDayDataGPU(VarInfoMapInputDeque, DayTleafDeque);
    ModelGetDayTleaf->Release();

    std::cout << "end DayTleafDeque" << std::endl;

    //////////////////////////////NPP/////////////////////////////
    std::cout << "begin NppDeque" << std::endl;

    std::deque<double *> *NppDeque = new std::deque<double *>;

    Pespiration *ModelPespiration = new Pespiration();
    ModelPespiration->Init(OutLaiInputDeque, SpatiotemporalDeque);
    ModelPespiration->Inneed();

    ModelPespiration->PespirationGPU(OutLaiInputDeque, DayTleafDeque, DayGppDeque, LandCoverInputDeque, NppDeque);
    // // /////////////write to tif////////////////////////////////
    const char *file = "../dataGdals/C3NPP";
    DataSet_Write *Datasetout = new DataSet_Write();
    Datasetout->ResultToTif(file, OutLaiInputDeque, NppDeque);

    ////////////Release////////////////////////////////////
    for (std::deque<double *>::iterator it = GppDeque3h->begin(); it != GppDeque3h->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = DayGppDeque->begin(); it != DayGppDeque->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = ClumpIndexInputDeque->begin(); it != ClumpIndexInputDeque->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = PercentC4InputDeque->begin(); it != PercentC4InputDeque->end(); it++)
    {
        delete[](*it);
    }
    */
    // SpatiotemporalInput->Release();
    // p_DataManager->Release();
    // ModelPsn->Release();
    // ModelGetDayGpp->Release();
    // ModelGetDayTleaf->Release();
    // ModelPespiration->Release();

    GDALDestroyDriverManager();
    printf("Done\n");

    return 0;
}
