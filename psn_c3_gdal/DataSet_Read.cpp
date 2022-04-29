
#include "DataSet_Read_Write.h"
#include "StructHeader.h"
using namespace std;


void DataSet_Read::ReadTif(const char* FilePath, std::deque<double*>* DatasetInput)
{
    Tif_Read* ClassDataset = new Tif_Read();
    ClassDataset->Init(FilePath);
    ClassDataset->ReadTifRaster();
    ClassDataset->ReadDataset();
    ClassDataset->TransDataset(DatasetInput);
    //ClassDataset->ReleaseDataset();
    std::cout << "DatasetInput:" << DatasetInput->size() << std::endl;
}

void DataSet_Read::ReadLaiTif(const char* FilePath, std::deque<InformStruct*>* LaiInput)
{
    Tif_Read* ClassLaiData = new Tif_Read();
    int TempResolution = 24;
    int SpaResolution = 500;

    ClassLaiData->Init(FilePath);
    ClassLaiData->ReadTifRaster();
    ClassLaiData->ReadLaiDataset();
    ClassLaiData->ReadSpatiotemporal(TempResolution, SpaResolution);
    ClassLaiData->TransLaiDataset(LaiInput);
    //ClassLaiData->ReleaseLaiDataset();
    std::cout << "LaiInput:" << LaiInput->size() << std::endl;
}

void DataSet_Read::Init()
{
    //���ٿռ�
    
    PressureInputDeque = new std::deque<double*>;
    RadInputDeque = new std::deque<double*>;
    SHInputDeque = new std::deque<double*>;
    TempInputDeque = new std::deque<double*>;

    LaiInputDeque = new std::deque<InformStruct*>;
    DataSetReadDeque = new std::deque<DatasetInputStruct*>;

}

void DataSet_Read::Release()
{
    //Lai
    //for (std::deque<InformStruct*>::iterator it = LaiInputDeque->begin(); it != LaiInputDeque->end(); it++)
    //{
    //    delete[](*it)->LaiBand->p0;
    //    delete[](*it)->LaiBand->p1;
    //    delete[](*it)->LaiBand->p2;
    //    delete[](*it)->LaiBand->p3;
    //    delete[](*it)->LaiBand->p4;
    //    delete[](*it)->LaiBand->p5;
    //    
    //    delete[](*it)->LaiBand->Lat;
    //    delete[](*it)->LaiBand->Lon;

    //    delete[](*it)->TifRaster->GeoTransform;
    //    delete[](*it)->Spatiotemporal->timeSeries;

    //    delete(*it);
    //}

    //Pressure
    for (std::deque<double*>::iterator it = PressureInputDeque->begin(); it != PressureInputDeque->end(); it++)
    {
        CPLFree((*it));
    }

    //Rad
    for (std::deque<double*>::iterator it = RadInputDeque->begin(); it != RadInputDeque->end(); it++)
    {
        CPLFree((*it));
    }

    //SH
    for (std::deque<double*>::iterator it = SHInputDeque->begin(); it != SHInputDeque->end(); it++)
    {
        CPLFree((*it));
    }

    //Temp
    for (std::deque<double*>::iterator it = TempInputDeque->begin(); it != TempInputDeque->end(); it++)
    {
        CPLFree((*it));
    }

    //�������
    for (std::deque<DatasetInputStruct*>::iterator it = DataSetReadDeque->begin(); it != DataSetReadDeque->end(); it++)
    {
        CPLFree((*it)->Pressure);
        CPLFree((*it)->Rad);
        CPLFree((*it)->SH);
        CPLFree((*it)->Temp);

        delete (*it);
    }

    PressureInputDeque->erase(PressureInputDeque->begin(), PressureInputDeque->end());
    RadInputDeque->erase(RadInputDeque->begin(), RadInputDeque->end());
    SHInputDeque->erase(SHInputDeque->begin(), SHInputDeque->end());
    TempInputDeque->erase(TempInputDeque->begin(), TempInputDeque->end());

    LaiInputDeque->erase(LaiInputDeque->begin(), LaiInputDeque->end());
    DataSetReadDeque->erase(DataSetReadDeque->begin(), DataSetReadDeque->end());

    std::cout << "Release" << std::endl;

    //GDALDestroyDriverManager();
}

void DataSet_Read::ReadDataset()
{
    const char* filepath_Lai = "/data/users/wangsj/dataset/lai_365_bands.tif";
    const char* filepath_Pressure = "/data/users/wangsj/dataset/Pressure_year_365.tif";
    const char* filepath_Rad = "/data/users/wangsj/dataset/Rad_year_365.tif";
    const char* filepath_SH = "/data/users/wangsj/dataset/SH_year_365.tif";
    const char* filepath_Temp = "/data/users/wangsj/dataset/Temp_year_365.tif";

    //ReadLaiTif(filepath_Lai, LaiInputDeque);
    //ReadTif(filepath_Pressure, PressureInputDeque);
    //ReadTif(filepath_Rad, RadInputDeque);
    //ReadTif(filepath_SH, SHInputDeque);
    //ReadTif(filepath_Temp, TempInputDeque);

    //���̼߳���
    std::thread lai(&DataSet_Read::ReadLaiTif, this, filepath_Lai, LaiInputDeque);
    std::thread pressure(&DataSet_Read::ReadTif, this, filepath_Pressure, PressureInputDeque);
    std::thread rad(&DataSet_Read::ReadTif, this, filepath_Rad, RadInputDeque);
    std::thread sh(&DataSet_Read::ReadTif, this, filepath_SH, SHInputDeque);
    std::thread temp(&DataSet_Read::ReadTif, this, filepath_Temp, TempInputDeque);

    std::cout << "lai Thread ID:" << lai.get_id() << std::endl;
    std::cout << "pressure Thread ID:" << pressure.get_id() << std::endl;
    std::cout << "rad Thread ID:" << rad.get_id() << std::endl;
    std::cout << "sh Thread ID:" << sh.get_id() << std::endl;
    std::cout << "temp Thread ID:" << temp.get_id() << std::endl;

    lai.join();
    pressure.join();
    rad.join();
    sh.join();
    temp.join();

    std::cout << "deque size:" << LaiInputDeque->size() << "\t"
        << PressureInputDeque->size() << "\t"
        << RadInputDeque->size() << "\t"
        << SHInputDeque->size() << "\t"
        << TempInputDeque->size() << "\t" << std::endl;

//#pragma omp parallel for ordered
    for (int i = 0; i < TempInputDeque->size(); i++)
    {
        DataSetReadDeque->push_back(new DatasetInputStruct);
        DataSetReadDeque->back()->Pressure = PressureInputDeque->at(i);
        DataSetReadDeque->back()->Rad = RadInputDeque->at(i);
        DataSetReadDeque->back()->SH = SHInputDeque->at(i);
        DataSetReadDeque->back()->Temp = TempInputDeque->at(i);
    }
}

void DataSet_Read::TransDataset(std::deque<InformStruct*>*LaiDestinationDataset,
                                std::deque<DatasetInputStruct*>* DestinationDataset)
{
    *LaiDestinationDataset = *LaiInputDeque;
    *DestinationDataset = *DataSetReadDeque;

    std::cout << "Finish TrandDataset"<< LaiInputDeque->size() <<"\t"<< DataSetReadDeque->size() << std::endl;

}