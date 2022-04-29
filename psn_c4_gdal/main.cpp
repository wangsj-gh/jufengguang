#include <fstream>
#include "StructHeader.h"
// #include "PSN.cuh"
#include "PSN.h"

#include "Tif_Read.h"
#include "DataSet_Read_Write.h"
#include <iomanip>

using namespace std;

int main(void)
{
    GDALAllRegister();
    
///////////////////////////////////////////////////////////////////////////////////////////////
    std::deque<InformStruct*>* OutLaiInputDeque=new std::deque<InformStruct*>;
    std::deque<DatasetInputStruct*>* OutDatasetReadDeque=new std::deque<DatasetInputStruct*>;

    DataSet_Read* DatasetInput = new DataSet_Read();
    DatasetInput->Init();
    DatasetInput->ReadDataset();
    DatasetInput->TransDataset(OutLaiInputDeque, OutDatasetReadDeque);
    std::cout << "DatasetInput:" << OutLaiInputDeque->size() << "\t" << OutDatasetReadDeque->size() << std::endl;
///////////////////////////////////////////////

    std::deque<double*>* GppDeque = new std::deque<double*>;


    PSN* ModelPsn = new PSN();
    ModelPsn->Init(OutLaiInputDeque->back()->TifRaster);
    ModelPsn->Inneed();
    for (int i = 0; i < OutDatasetReadDeque->size(); i++)
    {
        double* GPP = (double*)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize
                                                        * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        ModelPsn->PSNcomputeGPU(OutLaiInputDeque->back()->TifRaster,
            OutLaiInputDeque->back()->LaiBand->Lat, OutLaiInputDeque->back()->LaiBand->Lon,

            OutDatasetReadDeque->at(i)->Pressure, OutDatasetReadDeque->at(i)->Rad, 
            OutDatasetReadDeque->at(i)->SH, OutDatasetReadDeque->at(i)->Temp,

            OutLaiInputDeque->back()->LaiBand->p0, OutLaiInputDeque->back()->LaiBand->p1, OutLaiInputDeque->back()->LaiBand->p2,
            OutLaiInputDeque->back()->LaiBand->p3, OutLaiInputDeque->back()->LaiBand->p4, OutLaiInputDeque->back()->LaiBand->p5,
            OutLaiInputDeque->back()->Spatiotemporal->timeSeries[i% OutLaiInputDeque->back()->Spatiotemporal->TimeSize],
            i+1,
            GPP);
        GppDeque->push_back(GPP);
    }
    std::cout << "GppDeque:" << GppDeque->size() << std::endl;
    /////////////write to tif////////////////////////////////
    const char* file = "/data/users/wangsj/dataset/GPP_C4_gdals2";

    DataSet_Write* Datasetout = new DataSet_Write();
    Datasetout->ResultToTif(file, OutLaiInputDeque, GppDeque);

    ////////////Release////////////////////////////////////

    for (std::deque<double*>::iterator it = GppDeque->begin(); it != GppDeque->end(); it++)
    {
        delete[](*it);
        
    }

    for (std::deque<InformStruct*>::iterator it = OutLaiInputDeque->begin(); it != OutLaiInputDeque->end();it++)
    {
        delete[](*it)->TifRaster->GeoTransform;
        delete[](*it)->Spatiotemporal->timeSeries;
        CPLFree((*it)->LaiBand->p0);
        CPLFree((*it)->LaiBand->p1);
        CPLFree((*it)->LaiBand->p2);
        CPLFree((*it)->LaiBand->p3);
        CPLFree((*it)->LaiBand->p4);
        CPLFree((*it)->LaiBand->p5);

        delete[](*it)->LaiBand->Lon;
        delete[](*it)->LaiBand->Lat;
        delete (*it);

    }

    for (std::deque<DatasetInputStruct*>::iterator it = OutDatasetReadDeque->begin(); it != OutDatasetReadDeque->end(); it++)
    {
        CPLFree((*it)->Pressure);
        CPLFree((*it)->Rad);
        CPLFree((*it)->SH);
        CPLFree((*it)->Temp);

        delete (*it);
    }

    GppDeque->erase(GppDeque->begin(), GppDeque->end());
    OutLaiInputDeque->erase(OutLaiInputDeque->begin(), OutLaiInputDeque->end());
    OutDatasetReadDeque->erase(OutDatasetReadDeque->begin(), OutDatasetReadDeque->end());

    //DatasetInput->Release();
    ModelPsn->Release();

    GDALDestroyDriverManager();
    printf("Done\n");
  
    return 0;
}



