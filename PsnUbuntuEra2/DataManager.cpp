#include "DataManager.h"

void DataManager::Init()
{
	p_FilesScan = new FilesScan();
	p_DataQueue = new std::deque<VarInfo *>;

	// p_NetCDF_Read = new NetCDF_Read();
	// p_NetCDF_Read->init();
}
void DataManager::Release()
{
	// p_NetCDF_Read->Release();
	// delete p_NetCDF_Read;

	//释放文件扫描类
	p_FilesScan->Release();
	delete p_FilesScan;

	//释放缓存队列，一般情况为空
	for (std::deque<VarInfo *>::iterator it = p_DataQueue->begin(); it != p_DataQueue->end(); ++it)
	{
		std::cout << (*it)->Name << " " << (*it)->LongName << " ";

		if ((*it)->IsReadData)
		{
			int Count = 0;
			for (std::deque<unsigned char *>::iterator index = (*it)->Data.begin(); index != (*it)->Data.end(); ++index)
			{
				delete[](*index);
				std::cout << " " << Count;
				Count++;
			}
			std::cout << "delete Data ";
		}

		delete[](*it)->DimsName;
		delete[](*it)->DimsSize;
		delete (*it);

		std::cout << "Release~" << std::endl;
	}

	std::cout << "deque p_DataQueue Release~" << std::endl;
	p_DataQueue->erase(p_DataQueue->begin(), p_DataQueue->end());
	delete p_DataQueue;
}

void DataManager::SetDataQueue(std::deque<VarInfo *> *p_VarInfoQueue)
{
	this->p_DataQueue = p_VarInfoQueue;
}

std::map<std::string, std::deque<VarInfo *> *> DataManager::GetDataMap()
{

	return this->VarInfoMap;
}

void DataManager::CollectFilesPath() // vector，多个文件夹
{
	p_FilesScan->ScanFiles("/data/users/wangsj/dataGdals/lai2007");

	FilesPath = p_FilesScan->GetFiles();
}

void DataManager::ShowFilesPath()
{
	for (unsigned int idx = 0; idx < FilesPath.size(); idx++)
	{
		std::cout << "第" << idx << "个文件" << FilesPath.at(idx) << std::endl;
	}
}

void DataManager::ReadFiles()
{
	for (unsigned int idx = 0; idx < FilesPath.size(); idx++)
	{
		// std::this_thread::sleep_for(std::chrono::milliseconds(100));
		std::cout << "正在读取：第" << idx << "个文件" << FilesPath.at(idx) << std::endl;
		ReadData(FilesPath.at(idx));
		GenerateDataMap();
	}
	std::cout << "finish read data" << std::endl;
}

void DataManager::ReadData(std::string filepath)
{

	NetCDF_Read *p_NetCDF_Read = new NetCDF_Read();
	p_NetCDF_Read->init();

	p_NetCDF_Read->OpenFile(filepath);

	p_NetCDF_Read->ReadDims();
	p_NetCDF_Read->ReadVars();

	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

	// p_NetCDF_Read->ShowAllVars();
	// p_NetCDF_Read->ReadTemp();
	p_NetCDF_Read->ReadGeoCoordinates();
	// p_NetCDF_Read->ReadData("t2m");
	p_NetCDF_Read->ReadCollectedData(0);

	//数据输出
	p_NetCDF_Read->TransData(p_DataQueue);
	std::cout << "数据序列长度 " << p_DataQueue->size() << "数据传输完成... " << std::endl;

	p_NetCDF_Read->CloseFile();

	// p_NetCDF_Read->Release();
	// delete p_NetCDF_Read;
}

void DataManager::GenerateDataMap()
{
	std::map<std::string, std::deque<VarInfo *> *>::iterator it_map;

	while (!p_DataQueue->empty())
	{
		std::string VarName = p_DataQueue->front()->Name;
		// std::cout << "DataMap: " << VarName << "(" << p_DataQueue->front()->LongName << ")";

		it_map = VarInfoMap.find(VarName);

		if (it_map == VarInfoMap.end())
		{
			std::cout << "No VarMap~ ";
			VarInfoMap.insert(it_map, std::pair<std::string, std::deque<VarInfo *> *>(VarName, new std::deque<VarInfo *>));
			std::cout << "Generate VarMap: " << VarName << std::endl;
		}
		else
		{
			// std::cout << "VarMap existence" << std::endl;
		}

		VarInfoMap.find(VarName)->second->push_back(p_DataQueue->front());
		p_DataQueue->pop_front();

		// std::cout << p_DataQueue->size() << " " << VarInfoMap.find(VarName)->second->size() << " " << VarInfoMap.find(VarName)->second->back()->Data.size() << std::endl;
	}
}