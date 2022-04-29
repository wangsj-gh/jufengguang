#include <iostream>
#include <gdal.h>
#include <gdal_priv.h>
#include "StructHeader.h"
#include "Tif_Read.h"
using namespace std;


void Tif_Read::Init(const char* FilePath)
{
    // 注册
    //GDALAllRegister();
    // 创建一个类并打开文件
    poDataset = (GDALDataset*)GDALOpen(FilePath, GA_ReadOnly);

    // 判断文件是否为空
    if (poDataset == NULL )
    {
        std::cout << "Open failed.\n" << std::endl;
        GDALDestroyDriverManager();   
    }
    else { std::cout << "Open success.\n" << std::endl; }

    SpatiotemporalDeque = new std::deque<SpatiotemporalStruct*>;
    TifRasterDeque = new std::deque<TifRasterStruct*>;
    LaiDatasetDeque = new std::deque<LaiBandStruct*>;
    DatasetDeque = new std::deque<double*>;
}

void Tif_Read::ReleaseDataset()
{
    //TifRasterStruct
    for (std::deque<TifRasterStruct*>::iterator it = TifRasterDeque->begin(); it != TifRasterDeque->end(); it++)
    {
        delete[](*it)->GeoTransform;
        delete (*it);
    }

    //Dataset
    for (std::deque<double*>::iterator it = DatasetDeque->begin(); it != DatasetDeque->end(); it++)
    {
        CPLFree((*it));
        //delete (*it);
    }

    TifRasterDeque->erase(TifRasterDeque->begin(), TifRasterDeque->end());
    DatasetDeque->erase(DatasetDeque->begin(), DatasetDeque->end());

    GDALClose((GDALDatasetH)poDataset);
    std::cout << "ReleaseDataset Finish" << std::endl;
}

void Tif_Read::ReleaseLaiDataset()
{
    //SpatiotemporalDeque
    for (std::deque<SpatiotemporalStruct*>::iterator it = SpatiotemporalDeque->begin(); it != SpatiotemporalDeque->end(); it++)
    {
        delete[](*it)->timeSeries;
        delete (*it);
    }

    //TifRasterStruct
    for (std::deque<TifRasterStruct*>::iterator it = TifRasterDeque->begin(); it != TifRasterDeque->end(); it++)
    {
        delete[](*it)->GeoTransform;
        delete (*it);
    }

    //LaiBandStruct
    for (std::deque<LaiBandStruct*>::iterator it = LaiDatasetDeque->begin(); it != LaiDatasetDeque->end(); it++)
    {
        CPLFree((*it)->p0);
        CPLFree((*it)->p1);
        CPLFree((*it)->p2);
        CPLFree((*it)->p3);
        CPLFree((*it)->p4);
        CPLFree((*it)->p5);

        delete[](*it)->Lon;
        delete[](*it)->Lat;
        delete (*it);
    }

    SpatiotemporalDeque->erase(SpatiotemporalDeque->begin(), SpatiotemporalDeque->end());
    TifRasterDeque->erase(TifRasterDeque->begin(), TifRasterDeque->end());
    LaiDatasetDeque->erase(LaiDatasetDeque->begin(), LaiDatasetDeque->end());

    GDALClose((GDALDatasetH)poDataset);
    std::cout << "ReleaseLaiDataset Finish" << std::endl;
}


void Tif_Read::ReadTifRaster()
{
    //输出图像大小和波段个数
    double addGeoTransform[6];
    poDataset->GetGeoTransform(addGeoTransform);

    TifRasterDeque->push_back(new TifRasterStruct);
    TifRasterDeque->back()->RasterCount= poDataset->GetRasterCount();
    TifRasterDeque->back()->RasterXSize= poDataset->GetRasterXSize();
    TifRasterDeque->back()->RasterYSize= poDataset->GetRasterYSize();
    TifRasterDeque->back()->Projection = poDataset->GetProjectionRef();
    TifRasterDeque->back()->GeoTransform = new double[6];

    for (int i = 0; i < 6; i++)
    {
        TifRasterDeque->back()->GeoTransform[i] = addGeoTransform[i];
    }

    //打印X,Y,Count,GeoTransform

    //std::cout << "XSize*YSize*Count(Pressure):" << TifRasterDeque->back()->RasterXSize << "*"
    //    << TifRasterDeque->back()->RasterYSize << "*" << TifRasterDeque->back()->RasterCount << std::endl;
    //std::cout << "Total pixels(Pressure):" << TifRasterDeque->back()->RasterXSize * TifRasterDeque->back()->RasterYSize << std::endl;
    //std::cout << "GeoTransform:";
    //for (int i = 0; i < 6; i++) {
    //    std::cout << TifRasterDeque->back()->GeoTransform[i] << "   ";
    //}
    //std::cout << std::endl;
}

void Tif_Read::ReadLaiDataset()
{
    //the parameter of LAI
    GDALRasterBand* poBand;
    double* p0 = 0;
    double* p1 = 0;
    double* p2 = 0;
    double* p3 = 0;
    double* p4 = 0;
    double* p5 = 0;
    double* BandsList[] = { p0,p1,p2,p3,p4,p5 };
    for (int i = 1; i <= TifRasterDeque->back()->RasterCount; i++)
    {
        
        poBand = poDataset->GetRasterBand(i);
        BandsList[i-1] = (double*)CPLMalloc(sizeof(double) * TifRasterDeque->back()->RasterXSize *
                                                             TifRasterDeque->back()->RasterYSize);
        poBand->RasterIO(GF_Read, 0, 0, poBand->GetXSize(), poBand->GetYSize(),
                      BandsList[i - 1], poBand->GetXSize(), poBand->GetYSize(), GDT_Float64, 0, 0);
    }
    LaiDatasetDeque->push_back(new LaiBandStruct);
    LaiDatasetDeque->back()->p0 = BandsList[0];
    LaiDatasetDeque->back()->p1 = BandsList[1];
    LaiDatasetDeque->back()->p2 = BandsList[2];
    LaiDatasetDeque->back()->p3 = BandsList[3];
    LaiDatasetDeque->back()->p4 = BandsList[4];
    LaiDatasetDeque->back()->p5 = BandsList[5];

    //生成经纬度
    LaiDatasetDeque->back()->Lon = new double[TifRasterDeque->back()->RasterXSize * TifRasterDeque->back()->RasterYSize];
    LaiDatasetDeque->back()->Lat = new double[TifRasterDeque->back()->RasterXSize * TifRasterDeque->back()->RasterYSize];

    for (int i = 0; i < TifRasterDeque->back()->RasterYSize; i++)
    {
        for (int j = 0; j < TifRasterDeque->back()->RasterXSize; j++)
        {
            LaiDatasetDeque->back()->Lon[i * TifRasterDeque->back()->RasterXSize + j] =
                TifRasterDeque->back()->GeoTransform[0] +
                j * TifRasterDeque->back()->GeoTransform[1] +
                i * TifRasterDeque->back()->GeoTransform[2];
            LaiDatasetDeque->back()->Lat[i * TifRasterDeque->back()->RasterXSize + j] =
                TifRasterDeque->back()->GeoTransform[3] +
                j * TifRasterDeque->back()->GeoTransform[4] +
                i * TifRasterDeque->back()->GeoTransform[5];
        }
    }

    std::cout << "ReadLaiDataset Finish" << std::endl;

    //打印经纬度信息
  /*  for (std::deque<LaiBandStruct*>::iterator it = LaiDatasetDeque->begin(); it != LaiDatasetDeque->end(); it++)
    {
        for (int i = 0; i <  TifRasterDeque->back()->RasterXSize* TifRasterDeque->back()->RasterYSize; i++)
        {
            std::cout << "i=" <<i<< "\t" << (*it)->p0[i] << std::endl;
        }

    }*/
}


void Tif_Read::ReadDataset()
{
    GDALRasterBand* poBand;

//#pragma omp parallel for ordered //num_threads(20)
    for (int i = 1; i <= TifRasterDeque->back()->RasterCount; i++)
    { 
        poBand = poDataset->GetRasterBand(i);
        double* Data = (double*)CPLMalloc(sizeof(double) * TifRasterDeque->back()->RasterXSize * 
                                                            TifRasterDeque->back()->RasterYSize);
        poBand->RasterIO(GF_Read, 0, 0, poBand->GetXSize(), poBand->GetYSize(),
                                   Data, poBand->GetXSize(), poBand->GetYSize(), GDT_Float64, 0, 0);
        DatasetDeque->push_back(Data);       
    }
    std::cout << "ReadDataset Finish" << std::endl;
}

void Tif_Read::TransDataset(std::deque<double*>* DestinationData)
{
    *DestinationData = *DatasetDeque;
    std::cout << "TransDataset:" << DestinationData->size() << "\t" << DatasetDeque->size() << std::endl;
}

void Tif_Read::TransLaiDataset(std::deque<InformStruct*>* DestinationLai)
{
    DestinationLai->push_back(new InformStruct);
    DestinationLai->back()->LaiBand = LaiDatasetDeque->back();
    DestinationLai->back()->Spatiotemporal = SpatiotemporalDeque->back();
    DestinationLai->back()->TifRaster = TifRasterDeque->back();

    std::cout << "TransLaiDataset:" << DestinationLai->size() << "\t" << LaiDatasetDeque->size()
        << "\t" << SpatiotemporalDeque->size()
        << "\t" << TifRasterDeque->size()
        << std::endl;

    //DestinationLai->back()->LaiBand->Lat = LaiDatasetDeque->back()->Lat;
    //DestinationLai->back()->LaiBand->Lon = LaiDatasetDeque->back()->Lon;
    //DestinationLai->back()->LaiBand->p0 = LaiDatasetDeque->back()->p0;
    //DestinationLai->back()->LaiBand->p1 = LaiDatasetDeque->back()->p1;
    //DestinationLai->back()->LaiBand->p2 = LaiDatasetDeque->back()->p2;
    //DestinationLai->back()->LaiBand->p3 = LaiDatasetDeque->back()->p3;
    //DestinationLai->back()->LaiBand->p4 = LaiDatasetDeque->back()->p4;
    //DestinationLai->back()->LaiBand->p5 = LaiDatasetDeque->back()->p5;
    //             
    //DestinationLai->back()->Spatiotemporal->SpaResolution = SpatiotemporalDeque->back()->SpaResolution;
    //DestinationLai->back()->Spatiotemporal->TempResolution = SpatiotemporalDeque->back()->TempResolution;
    //DestinationLai->back()->Spatiotemporal->timeSeries = SpatiotemporalDeque->back()->timeSeries;
    //DestinationLai->back()->Spatiotemporal->TimeSize = SpatiotemporalDeque->back()->TimeSize;
    //             
    //DestinationLai->back()->TifRaster->GeoTransform = TifRasterDeque->back()->GeoTransform;
    //DestinationLai->back()->TifRaster->RasterCount = TifRasterDeque->back()->RasterCount;
    //DestinationLai->back()->TifRaster->RasterXSize = TifRasterDeque->back()->RasterXSize;
    //DestinationLai->back()->TifRaster->RasterYSize = TifRasterDeque->back()->RasterYSize;
}

void Tif_Read::ReadSpatiotemporal(int TempResolution, int SpaResolution)
{

    SpatiotemporalDeque->push_back(new SpatiotemporalStruct);
    SpatiotemporalDeque->back()->TempResolution = TempResolution;
    SpatiotemporalDeque->back()->SpaResolution = SpaResolution;
    SpatiotemporalDeque->back()->TimeSize = 24 / TempResolution;
    SpatiotemporalDeque->back()->timeSeries = new double[SpatiotemporalDeque->back()->TimeSize];

    if (SpatiotemporalDeque->back()->TimeSize == 1)
    {
        SpatiotemporalDeque->back()->timeSeries[0] = (double)TempResolution / 2;
    }
    else
    {
        SpatiotemporalDeque->back()->timeSeries[0] = (double)TempResolution / 2;
        for (int i = 1; i < SpatiotemporalDeque->back()->TimeSize; i++)
        {
            SpatiotemporalDeque->back()->timeSeries[i] = SpatiotemporalDeque->back()->timeSeries[i - 1] + (double)TempResolution;
        }
    }
}