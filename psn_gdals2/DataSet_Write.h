
#pragma once

#include "StructHeader.h"
#include "Tif_Read.h"

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