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
    int TempResolution = 1;
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
                     PercentC4InputDeque, VarInfoMapInputDeque, LandCoverInputDeque,
                     GppDeque3h);
    ModelPsn->Release();

    std::cout << "end PSN" << std::endl;

    const char *file1 = "../../dataEra/Gpp3hera";
    // const char *file1 = "../../dataEra/RHera";
    // const char *file1 = "/disknvme/raid00/wangsjnvme/Radera.tif";
    DataSet_Write *Dataset3h = new DataSet_Write();
    Dataset3h->ResultToTif(file1, OutLaiInputDeque, GppDeque3h);

    clock_t end2 = clock();
    std::cout << "GPU time:" << (double)(end2 - end1) / CLOCKS_PER_SEC << std::endl;

/*
    ////////////////get Day/////////////////////
    // std::cout << "begin DayDeque" << std::endl;

    // std::deque<double *> *DayDeque = new std::deque<double *>;

    // GetDayData *ModelDay = new GetDayData();
    // ModelDay->Init(OutLaiInputDeque, SpatiotemporalDeque);
    // ModelDay->Inneed();
    // ModelDay->GetDayDataGPU(GppDeque3h, DayDeque);
    // ModelDay->Release();

    // std::cout << "end DayDeque" << std::endl;

    // // const char *file2 = "../../dataEra/GppDayera";  
    // // const char *file2 = "../../dataEra/RHera";
    // DataSet_Write *Datasetday = new DataSet_Write();
    // Datasetday->ResultToTif(file2, OutLaiInputDeque, DayDeque);
    // clock_t end3 = clock();
    // std::cout << "write time:" << (double)(end3 - end2) / CLOCKS_PER_SEC << std::endl;


 
    ////////////////get GppDay/////////////////////
    std::cout << "begin GppDayDeque" << std::endl;

    std::deque<double *> *GppDayDeque = new std::deque<double *>;

    GetDayData *ModelDayGpp = new GetDayData();
    ModelDayGpp->Init(OutLaiInputDeque, SpatiotemporalDeque);
    ModelDayGpp->Inneed();
    ModelDayGpp->GetDayDataGPU(GppDeque3h, OutLaiInputDeque,SpatiotemporalDeque, GppDayDeque);
    ModelDayGpp->Release();

    std::cout << "end GppDayDeque" << std::endl;

    const char *file2 = "../../dataEra/GppDayera";  
    DataSet_Write *Datasetday = new DataSet_Write();
    Datasetday->ResultToTif(file2, OutLaiInputDeque, GppDayDeque);
    clock_t end3 = clock();
    std::cout << "write time:" << (double)(end3 - end2) / CLOCKS_PER_SEC << std::endl;



    //////////////get year data/////////////////////
    std::cout << "begin GetYearDeque" << std::endl;

    std::deque<double *> *YearGppDeque = new std::deque<double *>;

    GetDayData *ModelyearGpp = new GetDayData();
    ModelyearGpp->Init(OutLaiInputDeque, SpatiotemporalDeque);
    ModelyearGpp->Inneed();
    ModelyearGpp->GetYearDataGPU(GppDayDeque, YearGppDeque);
    // ModelyearGpp->GetYearDataGPU(GppDeque3h, YearGppDeque);
    ModelyearGpp->Release();

    std::cout << "end GetYearDeque" << std::endl;

    const char *file3 = "../../dataEra/GppYearera";
    // const char *file3 = "../../dataGdals/sunAcPrecent";
    DataSet_Write *Datasetyear = new DataSet_Write();
    Datasetyear->ResultToTif(file3, OutLaiInputDeque, YearGppDeque);

    clock_t end4 = clock();
    std::cout << "write GetYearDeque:" << (double)(end4 - end3) / CLOCKS_PER_SEC << std::endl;


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

    ModelPespiration->PespirationGPU(OutLaiInputDeque, DayTleafDeque, GppDayDeque, LandCoverInputDeque, NppDeque);
    // // /////////////write to tif////////////////////////////////
    const char *file = "../../dataEra/NppYearera";
    DataSet_Write *Datasetout = new DataSet_Write();
    Datasetout->ResultToTif(file, OutLaiInputDeque, NppDeque);

    ////////////Release////////////////////////////////////
    for (std::deque<double *>::iterator it = GppDeque3h->begin(); it != GppDeque3h->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = GppDayDeque->begin(); it != GppDayDeque->end(); it++)
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
    for (std::deque<double *>::iterator it = LandCoverInputDeque->begin(); it != LandCoverInputDeque->end(); it++)
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
