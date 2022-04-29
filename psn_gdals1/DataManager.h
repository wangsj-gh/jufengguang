#pragma once

#include <map>

#include"FilesScan.h"
#include "./netcdf/cxx4/netcdfcpp.h"
#include "NetCDF_Read.h"
#include <thread>
// #include <unistd.h>
//#include "ShowBands.h"

using namespace netCDF;
using namespace std;

class DataManager
{
public:
	void Init();
	void Release();
	
	void SetDataQueue(std::deque<VarInfo*>* p_VarInfoQueue);
	std::map<std::string, std::deque<VarInfo*>*> GetDataMap();
	void GenerateDataMap();
	//----------------------------------//
	

	//根据文件名，分类 变量
	//变量----queue


	//读取文件
	//文件----Netcdf类对象


	//输出数据队列

	//----------------------------------//
	//简要设计：依次读取多有文件，每个文件一个队列，不区分对象名称

	void CollectFilesPath();
	void ShowFilesPath();


	void ReadFiles();
	void ReadData(std::string filepath);


private:

	//存放文件夹下，所有文件的绝对路径
	std::vector<std::string> FilesPath;

	//std::vector<NetCDF_Read*> NetCDF_Vector;

	FilesScan* p_FilesScan;
	//NetCDF_Read* p_NetCDF_Read;

	std::deque<VarInfo*>* p_DataQueue;
	std::map<std::string, std::deque<VarInfo*>*> VarInfoMap;
};