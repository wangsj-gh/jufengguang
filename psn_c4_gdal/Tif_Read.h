#pragma once
#include <iostream>
#include <gdal.h>
#include <gdal_priv.h>
#include "StructHeader.h"

#include <deque>

using namespace std;

class Tif_Read
{
public:
	void Init(const char* FilePath);
	
	void ReadTifRaster();

	void ReadLaiDataset();
	void ReadSpatiotemporal(int TempResolution, int SpaResolution);

	void ReadDataset();

	void TransDataset(std::deque<double*>* DestinationData);
	void TransLaiDataset(std::deque<InformStruct*>* DestinationLai);

	void ReleaseDataset();
	void ReleaseLaiDataset();

private:
	GDALDataset* poDataset;
	const char* FilePath;

	std::deque<SpatiotemporalStruct*>* SpatiotemporalDeque;
	std::deque<TifRasterStruct*>* TifRasterDeque;
	std::deque<LaiBandStruct*>* LaiDatasetDeque;
	std::deque<double*>* DatasetDeque;
};