#include "StructHeader.h"
#include "DataSet_Read.h"
#include "DataSet_Write.h"
//the result GPP tif
void DataSet_Write::ResultToTif(const char* fileName,
                                std::deque<InformStruct*>* RasterDeque,
                                std::deque<double*>* DataDeque)
{
    GDALDataset* poDstDS;
    GDALDriver* poDriver;
    GDALRasterBand* po_Band;

    const char* pszFormat = "GTiff";
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);

    char** papszOptions = NULL;
    poDstDS = poDriver->Create(fileName,
                        RasterDeque->back()->TifRaster->RasterXSize,
                        RasterDeque->back()->TifRaster->RasterYSize,
                        DataDeque->size(),
                        GDT_Float64,
                        papszOptions);

    poDstDS->SetGeoTransform(RasterDeque->back()->TifRaster->GeoTransform);
    poDstDS->SetProjection(RasterDeque->back()->TifRaster->Projection);


    for (int i = 0; i < DataDeque->size(); i++)
    {

        po_Band = poDstDS->GetRasterBand(i + 1);
        po_Band->RasterIO(GF_Write, 0, 0, RasterDeque->back()->TifRaster->RasterXSize, RasterDeque->back()->TifRaster->RasterYSize,
            DataDeque->at(i), RasterDeque->back()->TifRaster->RasterXSize, RasterDeque->back()->TifRaster->RasterYSize, GDT_Float64, 0, 0);
    }
 

    GDALClose((GDALDatasetH)poDstDS);
}
