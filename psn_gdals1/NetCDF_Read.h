#pragma once

#include <iostream>
#include <sstream>
#include <map>
#include <queue>          // std::queue

#include"StructHeader.h"
#include "./netcdf/cxx4/netcdfcpp.h"

using namespace std;
using namespace netCDF;
// using namespace netCDF::exceptions;

class NetCDF_Read
{
public:
	void init();
	void OpenFile(const std::string filepath);
	void CloseFile();

	void Release();

	void ReadDims();
	void ReadVars();

	void ShowAllVars();
	void ReadTemp();
	void SetFile(std::string filepath);

	void ReadGeoCoordinates();
	template<typename T> void ReadOneDimData(std::string Name, VarInfo* it);

	void ReadData(std::string VarName, int BandRead);
	void ReadCollectedData(int BandRead = 0);

	void TransData(std::deque<VarInfo*> *Destination);

	std::string Convert(float Num);

private:

	NcFile* dataFile;

	int ReadDataDims;

	std::multimap<std::string, NcDim> group_dim;
	std::multimap<std::string, NcVar> group_var;
	std::deque<VarInfo*> VarInfoQueue;
};

