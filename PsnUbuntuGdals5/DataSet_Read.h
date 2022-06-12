#pragma once

#include "StructHeader.h"
#include "Tif_Read.h"

using namespace std;

class DataSet_Read
{
public:
	void Init();
	void ReadDataset();
	void VerifyData();
	void TransDataset(std::deque<InformStruct *> *LaiDestinationDataset,
					  std::deque<double *> *ClumpIndexDestinationDataset,
					  std::deque<double *> *PercentC4DestinationDataset,
					  std::deque<double *> *LandCoverDestinationDataset);
	void Release();

private:
	void ReadTif(const char *FilePath, std::deque<double *> *DestinationInput);
	void ReadLaiTif(const char *FilePath, std::deque<InformStruct *> *LaiInput);

	// std::deque<double *> *PressureInputDeque;
	// std::deque<double *> *RadInputDeque;
	// std::deque<double *> *SHInputDeque;
	// std::deque<double *> *TempInputDeque;

	std::deque<double *> *ClumpIndexInputDeque;
	std::deque<double *> *PercentC4InputDeque;
	std::deque<double *> *landCoverInputDeque;

	std::deque<InformStruct *> *LaiInputDeque;
	// std::deque<DatasetInputStruct *> *DataSetReadDeque;
};
