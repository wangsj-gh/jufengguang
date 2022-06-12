#include <iostream>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include "StructHeader.h"
#include "Tif_Read.h"
using namespace std;

void Tif_Read::Init(const char *FilePath)
{
    // ע��
    // GDALAllRegister();
    // ����һ���ಢ���ļ�
    poDataset = (GDALDataset *)GDALOpen(FilePath, GA_ReadOnly);

    // �ж��ļ��Ƿ�Ϊ��
    if (poDataset == NULL)
    {
        std::cout << "Open failed.\n"
                  << std::endl;
        GDALDestroyDriverManager();
    }
    else
    {
        std::cout << "Open success.\n"
                  << std::endl;
    }

    // SpatiotemporalDeque = new std::deque<SpatiotemporalStruct*>;
    TifRasterDeque = new std::deque<TifRasterStruct *>;
    // LaiDatasetDeque = new std::deque<LaiBandStruct*>;
    // DatasetDeque = new std::deque<double*>;
}

void Tif_Read::Release()
{
    // TifRasterStruct
    for (std::deque<TifRasterStruct *>::iterator it = TifRasterDeque->begin(); it != TifRasterDeque->end(); it++)
    {
        delete[](*it)->GeoTransform;
        delete[](*it)->Projection;
        delete (*it);
    }

    // Dataset
    //  for (std::deque<double*>::iterator it = DatasetDeque->begin(); it != DatasetDeque->end(); it++)
    //  {
    //      CPLFree((*it));
    //      //delete (*it);
    //  }

    TifRasterDeque->erase(TifRasterDeque->begin(), TifRasterDeque->end());
    // DatasetDeque->erase(DatasetDeque->begin(), DatasetDeque->end());

    GDALClose((GDALDatasetH)poDataset);
    std::cout << "ReleaseDataset Finish" << std::endl;
}

// void Tif_Read::ReleaseLaiDataset()
// {
//     //SpatiotemporalDeque
//     // for (std::deque<SpatiotemporalStruct*>::iterator it = SpatiotemporalDeque->begin(); it != SpatiotemporalDeque->end(); it++)
//     // {
//     //     delete[](*it)->timeSeries;
//     //     delete (*it);
//     // }

//     //TifRasterStruct
//     for (std::deque<TifRasterStruct*>::iterator it = TifRasterDeque->begin(); it != TifRasterDeque->end(); it++)
//     {
//         delete[](*it)->GeoTransform;
//         delete (*it);
//     }

//     //LaiBandStruct
//     for (std::deque<LaiBandStruct*>::iterator it = LaiDatasetDeque->begin(); it != LaiDatasetDeque->end(); it++)
//     {
//         CPLFree((*it)->p0);
//         CPLFree((*it)->p1);
//         CPLFree((*it)->p2);
//         CPLFree((*it)->p3);
//         CPLFree((*it)->p4);
//         CPLFree((*it)->p5);

//         delete[](*it)->Lon;
//         delete[](*it)->Lat;
//         delete (*it);
//     }

//     // SpatiotemporalDeque->erase(SpatiotemporalDeque->begin(), SpatiotemporalDeque->end());
//     TifRasterDeque->erase(TifRasterDeque->begin(), TifRasterDeque->end());
//     LaiDatasetDeque->erase(LaiDatasetDeque->begin(), LaiDatasetDeque->end());

//     GDALClose((GDALDatasetH)poDataset);
//     std::cout << "ReleaseLaiDataset Finish" << std::endl;
// }

void Tif_Read::ReadTifRaster()
{
    //���ͼ���С�Ͳ��θ���
    double addGeoTransform[6];
    poDataset->GetGeoTransform(addGeoTransform);

    TifRasterDeque->push_back(new TifRasterStruct);
    TifRasterDeque->back()->RasterCount = poDataset->GetRasterCount();
    TifRasterDeque->back()->RasterXSize = poDataset->GetRasterXSize();
    TifRasterDeque->back()->RasterYSize = poDataset->GetRasterYSize();
    TifRasterDeque->back()->Projection = poDataset->GetProjectionRef();
    TifRasterDeque->back()->GeoTransform = new double[6];

    for (int i = 0; i < 6; i++)
    {
        TifRasterDeque->back()->GeoTransform[i] = addGeoTransform[i];
    }

    //��ӡX,Y,Count,GeoTransform

    // std::cout << "XSize*YSize*Count(Pressure):" << TifRasterDeque->back()->RasterXSize << "*"
    //     << TifRasterDeque->back()->RasterYSize << "*" << TifRasterDeque->back()->RasterCount << std::endl;
    // std::cout << "Total pixels(Pressure):" << TifRasterDeque->back()->RasterXSize * TifRasterDeque->back()->RasterYSize << std::endl;
    // std::cout << "GeoTransform:";
    // for (int i = 0; i < 6; i++) {
    //     std::cout << TifRasterDeque->back()->GeoTransform[i] << "   ";
    // }
    // std::cout << std::endl;
}

void Tif_Read::ReadLaiDataset(std::deque<InformStruct *> &DestinationLai)
{
    // the parameter of LAI
    GDALRasterBand *poBand;
    double *p0 = NULL;
    double *p1 = NULL;
    double *p2 = NULL;
    double *p3 = NULL;
    double *p4 = NULL;
    double *p5 = NULL;
    double *BandsList[] = {p0, p1, p2, p3, p4, p5};
    for (int i = 1; i <= TifRasterDeque->back()->RasterCount; i++)
    {

        poBand = poDataset->GetRasterBand(i);
        BandsList[i - 1] = (double *)CPLMalloc(sizeof(double) * TifRasterDeque->back()->RasterXSize *
                                               TifRasterDeque->back()->RasterYSize);
        poBand->RasterIO(GF_Read, 0, 0, poBand->GetXSize(), poBand->GetYSize(),
                         BandsList[i - 1], poBand->GetXSize(), poBand->GetYSize(), GDT_Float64, 0, 0);
    }
    DestinationLai.push_back(new InformStruct);

    DestinationLai.back()->LaiBand = new LaiBandStruct();
    LaiBandStruct *LaiBand = DestinationLai.back()->LaiBand;

    LaiBand->p0 = BandsList[0];
    LaiBand->p1 = BandsList[1];
    LaiBand->p2 = BandsList[2];
    LaiBand->p3 = BandsList[3];
    LaiBand->p4 = BandsList[4];
    LaiBand->p5 = BandsList[5];

    //���ɾ�γ��
    LaiBand->Lon = new double[TifRasterDeque->back()->RasterXSize * TifRasterDeque->back()->RasterYSize];
    LaiBand->Lat = new double[TifRasterDeque->back()->RasterXSize * TifRasterDeque->back()->RasterYSize];

    for (int i = 0; i < TifRasterDeque->back()->RasterYSize; i++)
    {
        for (int j = 0; j < TifRasterDeque->back()->RasterXSize; j++)
        {
            LaiBand->Lon[i * TifRasterDeque->back()->RasterXSize + j] =
                TifRasterDeque->back()->GeoTransform[0] +
                j * TifRasterDeque->back()->GeoTransform[1] +
                i * TifRasterDeque->back()->GeoTransform[2];
            LaiBand->Lat[i * TifRasterDeque->back()->RasterXSize + j] =
                TifRasterDeque->back()->GeoTransform[3] +
                j * TifRasterDeque->back()->GeoTransform[4] +
                i * TifRasterDeque->back()->GeoTransform[5];
        }
    }

    DestinationLai.back()->TifRaster = new TifRasterStruct();
    memcpy(DestinationLai.back()->TifRaster, TifRasterDeque->back(), sizeof(TifRasterStruct)); //地址 copy

    DestinationLai.back()->TifRaster->GeoTransform = new double[6];
    memcpy(DestinationLai.back()->TifRaster->GeoTransform, TifRasterDeque->back()->GeoTransform, sizeof(double) * 6);

    std::cout << "ReadLaiDataset Finish" << std::endl;

    //��ӡ��γ����Ϣ
    /*  for (std::deque<LaiBandStruct*>::iterator it = LaiDatasetDeque->begin(); it != LaiDatasetDeque->end(); it++)
      {
          for (int i = 0; i <  TifRasterDeque->back()->RasterXSize* TifRasterDeque->back()->RasterYSize; i++)
          {
              std::cout << "i=" <<i<< "\t" << (*it)->p0[i] << std::endl;
          }

      }*/
}

void Tif_Read::ReadDataset(std::deque<double *> &DestinationData, int ReadBandCount)
{
    GDALRasterBand *poBand;
    int BandCount = (ReadBandCount == 0) ? TifRasterDeque->back()->RasterCount : ReadBandCount;
    int BandStart = 1;
    //#pragma omp parallel for ordered //num_threads(20)
    for (int i = BandStart; i <= BandCount; i++)
    {
        poBand = poDataset->GetRasterBand(i);
        double *Data = (double *)CPLMalloc(sizeof(double) * TifRasterDeque->back()->RasterXSize *
                                           TifRasterDeque->back()->RasterYSize);
        poBand->RasterIO(GF_Read, 0, 0, poBand->GetXSize(), poBand->GetYSize(),
                         Data, poBand->GetXSize(), poBand->GetYSize(), GDT_Float64, 0, 0);
        DestinationData.push_back(Data);
    }
    std::cout << "ReadDataset Finish" << std::endl;
}

// void Tif_Read::TransDataset(std::deque<double*>* DestinationData)
// {
//     *DestinationData = *DatasetDeque;
//     std::cout << "TransDataset:" << DestinationData->size() << "\t" << DatasetDeque->size() << std::endl;
// }

// void Tif_Read::TransLaiDataset(std::deque<InformStruct*>* DestinationLai)
// {
//     DestinationLai->push_back(new InformStruct);
//     DestinationLai->back()->LaiBand = LaiDatasetDeque->back();
//     // DestinationLai->back()->Spatiotemporal = SpatiotemporalDeque->back();
//     DestinationLai->back()->TifRaster = TifRasterDeque->back();

//     std::cout << "TransLaiDataset:" << DestinationLai->size() << "\t" << LaiDatasetDeque->size()
//         // << "\t" << SpatiotemporalDeque->size()
//         << "\t" << TifRasterDeque->size()
//         << std::endl;
// }

// void Tif_Read::ReadSpatiotemporal(int TempResolution, int SpaResolution)
// {

//     SpatiotemporalDeque->push_back(new SpatiotemporalStruct);
//     SpatiotemporalDeque->back()->TempResolution = TempResolution;
//     SpatiotemporalDeque->back()->SpaResolution = SpaResolution;
//     SpatiotemporalDeque->back()->TimeSize = 24 / TempResolution;
//     SpatiotemporalDeque->back()->timeSeries = new double[SpatiotemporalDeque->back()->TimeSize];

//     if (SpatiotemporalDeque->back()->TimeSize == 1)
//     {
//         SpatiotemporalDeque->back()->timeSeries[0] = (double)TempResolution / 2;
//     }
//     else
//     {
//         SpatiotemporalDeque->back()->timeSeries[0] = (double)TempResolution / 2;
//         for (int i = 1; i < SpatiotemporalDeque->back()->TimeSize; i++)
//         {
//             SpatiotemporalDeque->back()->timeSeries[i] = SpatiotemporalDeque->back()->timeSeries[i - 1] + (double)TempResolution;
//         }
//     }
// }