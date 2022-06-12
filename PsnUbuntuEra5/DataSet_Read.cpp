
#include "DataSet_Read.h"
#include "DataSet_Write.h"
#include "StructHeader.h"
using namespace std;

void DataSet_Read::ReadTif(const char *FilePath, std::deque<double *> *DatasetInput)
{
    Tif_Read *ClassDataset = new Tif_Read();
    ClassDataset->Init(FilePath);
    ClassDataset->ReadTifRaster();
    ClassDataset->ReadDataset(*DatasetInput, 0);
    // ClassDataset->TransDataset(DatasetInput);
    // ClassDataset->ReleaseDataset();
    //  ClassDataset->Release();
    std::cout << "DatasetInput:" << DatasetInput->size() << std::endl;
}

void DataSet_Read::ReadLaiTif(const char *FilePath, std::deque<InformStruct *> *LaiInput)
{
    Tif_Read *ClassLaiData = new Tif_Read();
    // int TempResolution = 24;
    // int SpaResolution = 500;

    ClassLaiData->Init(FilePath);
    ClassLaiData->ReadTifRaster();
    ClassLaiData->ReadLaiDataset(*LaiInput);
    // ClassLaiData->Release();
    // ClassLaiData->ReadSpatiotemporal(TempResolution, SpaResolution);
    // ClassLaiData->TransLaiDataset(LaiInput);
    // ClassLaiData->ReleaseLaiDataset();
    std::cout << "LaiInput:" << LaiInput->size() << std::endl;
}

void DataSet_Read::Init()
{
    //���ٿռ�

    // PressureInputDeque = new std::deque<double *>;
    // RadInputDeque = new std::deque<double *>;
    // SHInputDeque = new std::deque<double *>;
    // TempInputDeque = new std::deque<double *>;

    ClumpIndexInputDeque = new std::deque<double *>;
    PercentC4InputDeque = new std::deque<double *>;
    landCoverInputDeque = new std::deque<double *>;

    LaiInputDeque = new std::deque<InformStruct *>;

}

void DataSet_Read::Release()
{
    // Lai
    for (std::deque<InformStruct *>::iterator it = LaiInputDeque->begin(); it != LaiInputDeque->end(); it++)
    {
        delete[](*it)->LaiBand->p0;
        delete[](*it)->LaiBand->p1;
        delete[](*it)->LaiBand->p2;
        delete[](*it)->LaiBand->p3;
        delete[](*it)->LaiBand->p4;
        delete[](*it)->LaiBand->p5;

        delete[](*it)->LaiBand->Lat;
        delete[](*it)->LaiBand->Lon;
        delete[](*it)->LaiBand;

        delete[](*it)->TifRaster->GeoTransform;
        delete[](*it)->TifRaster->Projection;
        delete[](*it)->TifRaster;

        delete (*it);
    }

    LaiInputDeque->erase(LaiInputDeque->begin(), LaiInputDeque->end());


    std::cout << "Release" << std::endl;

    // GDALDestroyDriverManager();
}

void DataSet_Read::ReadDataset()
{
    const char *filepath_Lai = "/data/users/wangsj/dataEra/LaiParamOneCycle_2007_era5.tif";
    const char *filepath_ClumpIndex = "/data/users/wangsj/dataEra/CI_2007_era5.tif";
    const char *filepath_PercentC4 = "/data/users/wangsj/dataEra/c4_percent_1d_era5.tif";
    const char *filepath_LandCover = "/data/users/wangsj/dataEra/landcover_2007_era5.tif";

    std::thread lai(&DataSet_Read::ReadLaiTif, this, filepath_Lai, LaiInputDeque);
    std::thread ClumpIndex(&DataSet_Read::ReadTif, this, filepath_ClumpIndex, ClumpIndexInputDeque);
    std::thread PercentC4(&DataSet_Read::ReadTif, this, filepath_PercentC4, PercentC4InputDeque);
    std::thread Landcover(&DataSet_Read::ReadTif, this, filepath_LandCover, landCoverInputDeque);

    std::cout << "lai Thread ID:" << lai.get_id() << std::endl;
    std::cout << "ClumpIndex Thread ID:" << ClumpIndex.get_id() << std::endl;
    std::cout << "PercentC4 Thread ID:" << PercentC4.get_id() << std::endl;
    std::cout << "Landcover Thread ID:" << Landcover.get_id() << std::endl;

    lai.join();
    ClumpIndex.join();
    PercentC4.join();
    Landcover.join();
}

void DataSet_Read::VerifyData()
{
    std::cout << "LaiInputDeque size " << LaiInputDeque->size() << std::endl;
    std::cout << "ClumpIndexInputDeque size: " << ClumpIndexInputDeque->size() << std::endl;
    std::cout << "PercentC4InputDeque size: " << PercentC4InputDeque->size() << std::endl;
    std::cout << "landCoverInputDeque size: " << landCoverInputDeque->size() << std::endl;
}

void DataSet_Read::TransDataset(std::deque<InformStruct *> *LaiDestinationDataset,
                                std::deque<double *> *ClumpIndexDestinationDataset,
                                std::deque<double *> *PercentC4DestinationDataset,
                                std::deque<double *> *LandCoverDestinationDataset)
{
    // Lai
    size_t LaiQueueSize = LaiInputDeque->size();
    for (int i = 0; i < LaiQueueSize; i++)
    {
        LaiDestinationDataset->push_back(LaiInputDeque->front());
        LaiInputDeque->pop_front();

        std::cout << LaiInputDeque->size() << " " << LaiDestinationDataset->size() << std::endl;
    }
    std::cout << "Lai数据序列长度 " << LaiDestinationDataset->size() << " Lai数据传输完成... " << std::endl;

    // ClumpIndex
    ClumpIndexDestinationDataset->push_back(ClumpIndexInputDeque->front());

    // PercentC4
    PercentC4DestinationDataset->push_back(PercentC4InputDeque->front());

    // LandCover
    LandCoverDestinationDataset->push_back(landCoverInputDeque->front());

    std::cout << "Finish TrandDataset"<<'\n'
              << "Lai(size): " << LaiDestinationDataset->size() << "=" << LaiInputDeque->size() <<'\n'
              << "ClumpIndex(size): " << ClumpIndexDestinationDataset->size() << "=" << ClumpIndexInputDeque->size() <<'\n'
              << "PercentC4(size): " << PercentC4DestinationDataset->size() << "=" << PercentC4InputDeque->size() << '\n'
              << "LandCover(size): " << LandCoverDestinationDataset->size() << "=" << landCoverInputDeque->size() << '\n'
              << std::endl;
}