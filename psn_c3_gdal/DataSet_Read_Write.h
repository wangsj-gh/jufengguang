#pragma once

#include "StructHeader.h"
#include "Tif_Read.h"

using namespace std;

class DataSet_Read
{
public:
	//类内使用
	void ReadTif(const char* FilePath, std::deque<double*>* DestinationInput);
	void ReadLaiTif(const char* FilePath, std::deque<InformStruct*>* LaiInput);
	//外部调用
	void Init();
	void ReadDataset();
	void TransDataset(std::deque<InformStruct*>* LaiDestinationDataset,
					  std::deque<DatasetInputStruct*>* DestinationDataset);
	void Release();

private:
	std::deque<double*>* PressureInputDeque;
	std::deque<double*>* RadInputDeque;
	std::deque<double*>* SHInputDeque;
	std::deque<double*>* TempInputDeque;

	std::deque<InformStruct*>* LaiInputDeque;
	std::deque<DatasetInputStruct*>* DataSetReadDeque;
};

class DataSet_Write
{
public:
	void ResultToTif(const char* fileName,
					std::deque<InformStruct*>* RasterDeque,
					std::deque<double*>* DataDeque);
private:
	const char* fileName;
	std::deque<InformStruct*>* RasterDeque;
	std::deque<double*>* DataDeque;
};