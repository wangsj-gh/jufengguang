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
    printf("first");
    ///////////////////////////////////////////////////////////////////////////////////////////////
    std::deque<InformStruct *> *OutLaiInputDeque = new std::deque<InformStruct *>;
    std::deque<DatasetInputStruct *> *OutDatasetReadDeque = new std::deque<DatasetInputStruct *>;
    printf("second");

    DataSet_Read *DatasetInput = new DataSet_Read();
    DatasetInput->Init();
    DatasetInput->ReadDataset();
    DatasetInput->TransDataset(OutLaiInputDeque, OutDatasetReadDeque);
    std::cout << "DatasetInput:" << OutLaiInputDeque->size() << "\t" << OutDatasetReadDeque->size() << std::endl;
    ///////////////////////////////////////////////
    std::deque<double *> *Deque0 = new std::deque<double *>;
    std::deque<double *> *Deque1 = new std::deque<double *>;
    std::deque<double *> *Deque2 = new std::deque<double *>;
    std::deque<double *> *Deque3 = new std::deque<double *>;
    std::deque<double *> *Deque4 = new std::deque<double *>;
    std::deque<double *> *Deque5 = new std::deque<double *>;
    std::deque<double *> *Deque6 = new std::deque<double *>;
    std::deque<double *> *Deque7 = new std::deque<double *>;
    std::deque<double *> *Deque8 = new std::deque<double *>;
    std::deque<double *> *Deque9 = new std::deque<double *>;

    PSN *ModelPsn = new PSN();
    ModelPsn->Init(OutLaiInputDeque->back()->TifRaster);
    ModelPsn->Inneed();
    for (int i = 0; i < OutDatasetReadDeque->size(); i++)
    {
        double *var0 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var1 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var2 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var3 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var4 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var5 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var6 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var7 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var8 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);
        double *var9 = (double *)malloc(sizeof(double) * OutLaiInputDeque->back()->TifRaster->RasterXSize * OutLaiInputDeque->back()->TifRaster->RasterYSize);

        ModelPsn->PSNcomputeGPU(OutLaiInputDeque->back()->TifRaster,
                                OutLaiInputDeque->back()->LaiBand->Lat, OutLaiInputDeque->back()->LaiBand->Lon,

                                OutDatasetReadDeque->at(i)->Pressure, OutDatasetReadDeque->at(i)->Rad,
                                OutDatasetReadDeque->at(i)->SH, OutDatasetReadDeque->at(i)->Temp,

                                OutLaiInputDeque->back()->LaiBand->p0, OutLaiInputDeque->back()->LaiBand->p1, OutLaiInputDeque->back()->LaiBand->p2,
                                OutLaiInputDeque->back()->LaiBand->p3, OutLaiInputDeque->back()->LaiBand->p4, OutLaiInputDeque->back()->LaiBand->p5,
                                OutLaiInputDeque->back()->Spatiotemporal->timeSeries[i % OutLaiInputDeque->back()->Spatiotemporal->TimeSize],
                                i + 1,
                                var0, var1, var2, var3, var4, var5, var6, var7, var8, var9);

        Deque0->push_back(var0);
        Deque1->push_back(var1);
        Deque2->push_back(var2);
        Deque3->push_back(var3);
        Deque4->push_back(var4);
        Deque5->push_back(var5);
        Deque6->push_back(var6);
        Deque7->push_back(var7);
        Deque8->push_back(var8);
        Deque9->push_back(var9);
    }
    // std::cout << "var1ppDeque:" << var1ppDeque->size() << std::endl;
    /////////////write to tif////////////////////////////////
    // const char *file0 = "/data/users/wangsj/dataset/Lai";
    // DataSet_Write *Datasetoutvar0 = new DataSet_Write();
    // Datasetoutvar0->ResultToTif(file0, OutLaiInputDeque, Deque0);

    // const char *file1 = "/data/users/wangsj/dataset/VPD";
    // DataSet_Write *Datasetoutvar1 = new DataSet_Write();
    // Datasetoutvar1->ResultToTif(file1, OutLaiInputDeque, Deque1);

    // const char *file2 = "/data/users/wangsj/dataset/RH";
    // DataSet_Write *Datasetoutvar2 = new DataSet_Write();
    // Datasetoutvar2->ResultToTif(file2, OutLaiInputDeque, Deque2);

    // const char *file3 = "/data/users/wangsj/dataset/Jmax";
    // DataSet_Write *DatasetoutLai = new DataSet_Write();
    // DatasetoutLai->ResultToTif(file3, OutLaiInputDeque, Deque3);

    // const char *file4 = "/data/users/wangsj/dataset/tiemsers";
    // DataSet_Write *Datasetoutvar4 = new DataSet_Write();
    // Datasetoutvar4->ResultToTif(file4, OutLaiInputDeque, Deque4);

    // const char *file5 = "/data/users/wangsj/dataset/miukb";
    // DataSet_Write *Datasetoutvar5 = new DataSet_Write();
    // Datasetoutvar5->ResultToTif(file5, OutLaiInputDeque, Deque5);

    // const char *file6 = "/data/users/wangsj/dataset/G";
    // DataSet_Write *Datasetoutvar6 = new DataSet_Write();
    // Datasetoutvar6->ResultToTif(file6, OutLaiInputDeque, Deque6);

    // const char *file7 = "/data/users/wangsj/dataset/kb";
    // DataSet_Write *Datasetoutvar7 = new DataSet_Write();
    // Datasetoutvar7->ResultToTif(file7, OutLaiInputDeque, Deque7);

    // const char *file8 = "/data/users/wangsj/dataset/Rad_direct";
    // DataSet_Write *Datasetoutvar8 = new DataSet_Write();
    // Datasetoutvar8->ResultToTif(file8, OutLaiInputDeque, Deque8);

    // const char *file9 = "/data/users/wangsj/dataset/Rad_diffuse";
    // DataSet_Write *Datasetoutvar9 = new DataSet_Write();
    // Datasetoutvar9->ResultToTif(file9, OutLaiInputDeque, Deque9);

    // /////////////write to tif////////////////////////////////
    const char *file0 = "/data/users/wangsj/dataset/LAI_sun1";
    DataSet_Write *Datasetoutvar0 = new DataSet_Write();
    Datasetoutvar0->ResultToTif(file0, OutLaiInputDeque, Deque0);

    const char *file1 = "/data/users/wangsj/dataset/LAI_shade1";
    DataSet_Write *Datasetoutvar1 = new DataSet_Write();
    Datasetoutvar1->ResultToTif(file1, OutLaiInputDeque, Deque1);

    const char *file2 = "/data/users/wangsj/dataset/PPFD_sun1";
    DataSet_Write *Datasetoutvar2 = new DataSet_Write();
    Datasetoutvar2->ResultToTif(file2, OutLaiInputDeque, Deque2);

    const char *file3 = "/data/users/wangsj/dataset/PPFD_shade1";
    DataSet_Write *DatasetoutLai = new DataSet_Write();
    DatasetoutLai->ResultToTif(file3, OutLaiInputDeque, Deque3);

    const char *file4 = "/data/users/wangsj/dataset/A_sun1";
    DataSet_Write *Datasetoutvar4 = new DataSet_Write();
    Datasetoutvar4->ResultToTif(file4, OutLaiInputDeque, Deque4);

    const char *file5 = "/data/users/wangsj/dataset/A_shade1";
    DataSet_Write *Datasetoutvar5 = new DataSet_Write();
    Datasetoutvar5->ResultToTif(file5, OutLaiInputDeque, Deque5);
    /////////////////////////////////////////////////////////////////////////////////
    const char *file6 = "/data/users/wangsj/dataset/A_total1";
    DataSet_Write *Datasetoutvar6 = new DataSet_Write();
    Datasetoutvar6->ResultToTif(file6, OutLaiInputDeque, Deque6);
    ////////////////////////////////////////////////////////////////////////////////
    // const char *file6 = "/data/users/wangsj/dataset/taob";
    // DataSet_Write *Datasetoutvar6 = new DataSet_Write();
    // Datasetoutvar6->ResultToTif(file6, OutLaiInputDeque, Deque6);

    // const char *file7 = "/data/users/wangsj/dataset/betab";
    // DataSet_Write *Datasetoutvar7 = new DataSet_Write();
    // Datasetoutvar7->ResultToTif(file7, OutLaiInputDeque, Deque7);

    // const char *file8 = "/data/users/wangsj/dataset/taod";
    // DataSet_Write *Datasetoutvar8 = new DataSet_Write();
    // Datasetoutvar8->ResultToTif(file8, OutLaiInputDeque, Deque8);

    // const char *file9 = "/data/users/wangsj/dataset/betad";
    // DataSet_Write *Datasetoutvar9 = new DataSet_Write();
    // Datasetoutvar9->ResultToTif(file9, OutLaiInputDeque, Deque9);

    ////////////Release////////////////////////////////////

    for (std::deque<double *>::iterator it = Deque0->begin(); it != Deque0->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque1->begin(); it != Deque1->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque2->begin(); it != Deque2->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque3->begin(); it != Deque3->end(); it++)
    {
        delete[](*it);
    }

    for (std::deque<double *>::iterator it = Deque4->begin(); it != Deque4->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque5->begin(); it != Deque5->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque6->begin(); it != Deque6->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque7->begin(); it != Deque7->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque8->begin(); it != Deque8->end(); it++)
    {
        delete[](*it);
    }
    for (std::deque<double *>::iterator it = Deque9->begin(); it != Deque9->end(); it++)
    {
        delete[](*it);
    }

    for (std::deque<InformStruct *>::iterator it = OutLaiInputDeque->begin(); it != OutLaiInputDeque->end(); it++)
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

    for (std::deque<DatasetInputStruct *>::iterator it = OutDatasetReadDeque->begin(); it != OutDatasetReadDeque->end(); it++)
    {
        CPLFree((*it)->Pressure);
        CPLFree((*it)->Rad);
        CPLFree((*it)->SH);
        CPLFree((*it)->Temp);

        delete (*it);
    }
    Deque0->erase(Deque0->begin(), Deque0->end());
    Deque1->erase(Deque1->begin(), Deque1->end());
    Deque2->erase(Deque2->begin(), Deque2->end());
    Deque3->erase(Deque3->begin(), Deque3->end());
    Deque4->erase(Deque4->begin(), Deque4->end());
    Deque5->erase(Deque5->begin(), Deque5->end());
    Deque6->erase(Deque6->begin(), Deque6->end());
    Deque7->erase(Deque7->begin(), Deque7->end());
    Deque8->erase(Deque8->begin(), Deque8->end());
    Deque9->erase(Deque9->begin(), Deque9->end());
    OutLaiInputDeque->erase(OutLaiInputDeque->begin(), OutLaiInputDeque->end());
    OutDatasetReadDeque->erase(OutDatasetReadDeque->begin(), OutDatasetReadDeque->end());

    // DatasetInput->Release();
    ModelPsn->Release();

    GDALDestroyDriverManager();
    printf("Done\n");

    return 0;
}
