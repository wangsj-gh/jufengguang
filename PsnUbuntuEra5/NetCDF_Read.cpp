#include "NetCDF_Read.h"

void NetCDF_Read::init()
{
	dataFile = new NcFile();
}

void NetCDF_Read::OpenFile(const std::string filepath)
{
	dataFile->open(filepath, NcFile::read);
}

void NetCDF_Read::CloseFile()
{
	dataFile->close();
}

void NetCDF_Read::Release()
{
	for (std::deque<VarInfo *>::iterator it = VarInfoQueue.begin(); it != VarInfoQueue.end(); ++it)
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

	std::cout << "NetCDF_Read::Release~" << std::endl;
	VarInfoQueue.erase(VarInfoQueue.begin(), VarInfoQueue.end());

	group_dim.erase(group_dim.begin(), group_dim.end());
	group_var.erase(group_var.begin(), group_var.end());

	// dataFile->close();
	delete dataFile;
}

void NetCDF_Read::ReadDims()
{
	//维度操作
	group_dim = dataFile->getDims();

	std::multimap<std::string, NcDim>::iterator it_dim;
	it_dim = group_dim.begin();

	this->ReadDataDims = 0;

	for (; it_dim != group_dim.end(); it_dim++)
	{
		int id = it_dim->second.getId();

		std::string name = it_dim->second.getName();

		int size = it_dim->second.getSize();

		// std::cout << "ID:" << id << "  Name:" << name << "  Size:" << size << std::endl;

		this->ReadDataDims++;
	}

}

void NetCDF_Read::ReadVars()
{
	//变量操作
	group_var = dataFile->getVars();

	std::multimap<std::string, NcVar>::iterator it_var;
	it_var = group_var.begin();
	int row = 0;

	for (; it_var != group_var.end(); it_var++)
	{
		//读取变量ID
		int id = it_var->second.getId();

		//读取变量名称
		std::string name = it_var->second.getName();

		//读取变量类型
		std::string type = it_var->second.getType().getName();

		//读取变量所拥有的维度名称
		int dimCount = it_var->second.getDimCount();

		VarInfoQueue.push_back(new VarInfo());
		VarInfoQueue.back()->ID = id;
		VarInfoQueue.back()->Name = name;
		VarInfoQueue.back()->DataType = type;

		VarInfoQueue.back()->DimsCount = dimCount;
		VarInfoQueue.back()->DimsName = new std::string[dimCount];
		VarInfoQueue.back()->DimsSize = new int[dimCount];

		//读取变量所拥有的维度名称
		int AttCount = it_var->second.getAttCount();

		// std::cout << "ID:" << id << "  Name:" << name << "  Type:" << type << "  Dim:" << dimCount << " Attribute:" << AttCount << std::endl;

		std::string name_dim = "";
		for (int i = 0; i < it_var->second.getDimCount(); i++)
		{
			NcDim dim = it_var->second.getDim(i);
			if (i == 0)
			{
				name_dim = name_dim + dim.getName();
			}
			else
			{
				name_dim = name_dim + " , " + dim.getName();
			}

			VarInfoQueue.back()->DimsName[i] = dim.getName();
			VarInfoQueue.back()->DimsSize[i] = dim.getSize();
		}

		// std::cout << "name_dim:" << name_dim << std::endl;

		std::string name_attris = "";

		std::map<std::string, NcVarAtt> map_atts = it_var->second.getAtts();

		//查看压缩值
		float fscale = 1.0f;
		float foffset = 0.0f;

		if (map_atts.find("scale_factor") != map_atts.end())
		{
			NcVarAtt AttT = it_var->second.getAtt("scale_factor");
			if (!AttT.isNull())
			{
				NcType TypeId = AttT.getType();

				if (TypeId == NC_FLOAT || TypeId == NC_DOUBLE)
					AttT.getValues(&fscale);

				VarInfoQueue.back()->scale_factor = fscale;
				name_attris += "scale_factor:" + Convert(fscale) + " ";
			}
		}
		//查看偏移值
		if (map_atts.find("add_offset") != map_atts.end())
		{
			NcVarAtt AttS = it_var->second.getAtt("add_offset");
			if (!AttS.isNull())
			{
				NcType TypeId = AttS.getType();

				if (TypeId == NC_FLOAT || TypeId == NC_DOUBLE)
					AttS.getValues(&foffset);

				VarInfoQueue.back()->add_offset = foffset;
				name_attris += "add_offset:" + Convert(foffset) + " ";
			}
		}

		//默认填充值
		float fFillVaue = 0.0f;
		if (map_atts.find("_FillValue") != map_atts.end())
		{
			NcVarAtt AttF = it_var->second.getAtt("_FillValue");
			if (!AttF.isNull())
			{
				NcType TypeId = AttF.getType();

				if (TypeId == NC_FLOAT || TypeId == NC_DOUBLE || TypeId == NC_SHORT)
					AttF.getValues(&fFillVaue);

				VarInfoQueue.back()->fill_value = fFillVaue;
				name_attris += "_FillValue:" + Convert(fFillVaue) + " ";
			}
		}

		//无数据值
		float missing_value = 0.0f;
		if (map_atts.find("missing_value") != map_atts.end())
		{
			NcVarAtt AttF = it_var->second.getAtt("missing_value");
			if (!AttF.isNull())
			{
				NcType TypeId = AttF.getType();

				if (TypeId == NC_FLOAT || TypeId == NC_DOUBLE || TypeId == NC_SHORT)
					AttF.getValues(&missing_value);

				VarInfoQueue.back()->missing_value = missing_value;
				name_attris += "missing_value:" + Convert(missing_value) + "\n ";
			}
		}

		//完整名称
		// std::string long_name;
		if (map_atts.find("long_name") != map_atts.end())
		{
			NcVarAtt AttF = it_var->second.getAtt("long_name");
			if (!AttF.isNull())
			{
				char *long_name = new char[100];
				NcType TypeId = AttF.getType();

				// if (TypeId == NC_FLOAT || TypeId == NC_DOUBLE || TypeId == NC_SHORT)
				AttF.getValues(long_name);

				VarInfoQueue.back()->LongName = long_name;
				name_attris += "long_name:";
				name_attris += +long_name;
				name_attris += "\n ";
				delete[] long_name;
			}
		}

		// std::cout << name_attris << std::endl;

		row++;
	}
}

void NetCDF_Read::ShowAllVars()
{
	for (unsigned int index = 0; index < VarInfoQueue.size(); index++)
	{
		std::cout << "ID:" << VarInfoQueue.at(index)->ID << std::endl;
		std::cout << "Name:" << VarInfoQueue.at(index)->Name << std::endl;
		std::cout << "LongName:" << VarInfoQueue.at(index)->LongName << std::endl;
		std::cout << "DataType:" << VarInfoQueue.at(index)->DataType << std::endl;

		for (int i = 0; i < VarInfoQueue.at(index)->DimsCount; i++)
		{
			std::cout << "DimsName " << i << ": " << VarInfoQueue.at(index)->DimsName[i] << "Size \t" << VarInfoQueue.at(index)->DimsSize[i] << std::endl;
		}
		std::cout << std::endl;
	}
}

void NetCDF_Read::ReadTemp()
{
	std::cout << ".........读取数据: 温度" << std::endl;

	//读取数据
	NcVar mVar_d2m = dataFile->getVar("t2m");
	if (mVar_d2m.isNull())
		printf("error!\n\n");

	int nDims_d2m = mVar_d2m.getDimCount(); //维数

	// long long nCounts_d2m = 1;
	for (int i = 0; i < nDims_d2m; i++)
	{
		int nSize = mVar_d2m.getDim(i).getSize();
		// nCounts_d2m *= nSize;

		std::cout << "维度" << i << ":" << nSize << std::endl;
	}

	short *data_d2m = new short[1081 * 3600];

	std::vector<size_t> index(3);
	index[0] = 0;
	index[1] = 0;
	index[2] = 0;

	std::vector<size_t> index2(3);
	index2[0] = 1;
	index2[1] = 1000;
	index2[2] = 3600;

	mVar_d2m.getVar(index, index2, data_d2m);

}

void NetCDF_Read::SetFile(std::string filepath)
{
}

template <typename T>
void NetCDF_Read::ReadOneDimData(std::string Name, VarInfo *p_VarInfo)
{
	//读取数据
	NcVar mVar_Read = dataFile->getVar(Name);
	if (mVar_Read.isNull())
		printf("error!\n\n");

	std::vector<size_t> index1(1);
	std::vector<size_t> index2(1);

	index1[0] = 0;
	index2[0] = p_VarInfo->DimsSize[0];

	p_VarInfo->IsReadData = true;

	unsigned char *DataNew = (unsigned char *)malloc(p_VarInfo->DimsSize[0] * sizeof(T));
	mVar_Read.getVar(index1, index2, (T *)DataNew);

	p_VarInfo->Data.push_back(DataNew);
	// std::cout << "Read Complete " << p_VarInfo->Name << std::endl;

	//显示数据内容

	bool IsShowData = false;
	if (IsShowData)
	{
		for (size_t i = 0; i < p_VarInfo->DimsSize[0]; i++)
		{
			std::cout << "Value " << p_VarInfo->Name << " : " << i << ": " << ((T *)DataNew)[i] << std::endl;
		}
	}
}

void NetCDF_Read::ReadGeoCoordinates()
{
	// std::vector<size_t> index1(1);
	// std::vector<size_t> index2(1);

	int OneDimCount = 0;

	for (std::deque<VarInfo *>::iterator it = VarInfoQueue.begin(); it != VarInfoQueue.end(); ++it)
	{

		if ((*it)->DimsCount == 1)
		{
			OneDimCount++;

			// std::cout << ".........读取数据: " << (*it)->Name << std::endl;

			if ((*it)->DataType == "float")
			{
				ReadOneDimData<float>((*it)->Name, (*it));
			}

			if ((*it)->DataType == "int")
			{
				ReadOneDimData<int>((*it)->Name, (*it));
			}

			if ((*it)->DataType == "double")
			{
				ReadOneDimData<double>((*it)->Name, (*it));
			}
		}

		/*

		if ((*it)->LongName == "latitude")
		{
			std::cout << ".........读取数据: latitude" << std::endl;

			//读取数据
			NcVar mVar_latitude = dataFile->getVar("latitude");
			if (mVar_latitude.isNull())
				printf("error!\n\n");

			index1[0] = 0;
			index2[0] = (*it)->DimsSize[0];

			(*it)->IsReadData = true;
			unsigned char* DataNew = (unsigned char*)malloc((*it)->DimsSize[0]*sizeof(float));//new float[(*it)->DimsSize[0]];
			mVar_latitude.getVar(index1, index2, (float*)DataNew);

			(*it)->Data.push_back(DataNew);
			std::cout << "Read Latitude Complete" << std::endl;

			//显示数据内容
			for (size_t i = 0; i < (*it)->DimsSize[0]; i++)
			{
				std::cout << "Value Latitude: " << i << ": " << ((float*)DataNew)[i] << std::endl;
			}
		}

		if ((*it)->LongName == "longitude")
		{
			std::cout << ".........读取数据: longitude" << std::endl;

			//读取数据
			NcVar mVar_longitude = dataFile->getVar("longitude");
			if (mVar_longitude.isNull())
				printf("error!\n\n");

			index1[0] = 0;
			index2[0] = (*it)->DimsSize[0];

			(*it)->IsReadData = true;
			unsigned char* DataNew = (unsigned char*)malloc((*it)->DimsSize[0] * sizeof(float));//new float[(*it)->DimsSize[0]];
			mVar_longitude.getVar(index1, index2, (float*)DataNew);

			(*it)->Data.push_back(DataNew);
			std::cout << "Read longitude Complete" << std::endl;

			//显示数据内容
			for (size_t i = 0; i < (*it)->DimsSize[0]; i++)
			{
				std::cout << "Value longitude: " << i << ": " << ((float*)DataNew)[i] << std::endl;
			}
		}

		if ((*it)->LongName == "time")
		{
			std::cout << ".........读取数据: time" << std::endl;

			//读取数据
			NcVar mVar_time = dataFile->getVar("time");
			if (mVar_time.isNull())
				printf("error!\n\n");

			index1[0] = 0;
			index2[0] = (*it)->DimsSize[0];

			(*it)->IsReadData = true;
			unsigned char* DataNew = (unsigned char*)malloc((*it)->DimsSize[0] * sizeof(int)); //new float[(*it)->DimsSize[0]];
			mVar_time.getVar(index1, index2, (int*)DataNew);

			(*it)->Data.push_back(DataNew);
			std::cout << "Read time Complete" << std::endl;

			//显示数据内容
			for (size_t i = 0; i < (*it)->DimsSize[0]; i++)
			{
				std::cout << "Value time: " << i << ": " << ((int*)DataNew)[i] << std::endl;
			}
		}

		*/
	}

	// std::cout << "Read OneDimCount:" << OneDimCount << std::endl;
}

void NetCDF_Read::ReadData(std::string VarName, int BandRead)
{
	std::vector<size_t> index1(3);
	std::vector<size_t> index2(3);

	for (std::deque<VarInfo *>::iterator it = VarInfoQueue.begin(); it != VarInfoQueue.end(); ++it)
	{
		if ((*it)->Name == VarName)
		{
			// std::cout << ".........读取数据: " << VarName << std::endl;

			//读取数据
			NcVar mVar_temp = dataFile->getVar(VarName);
			if (mVar_temp.isNull())
				printf("error!\n\n");

			int BandsTimeSize = (BandRead == 0) ? (*it)->DimsSize[0] : BandRead;

			for (int BandsTime = 0; BandsTime < BandsTimeSize; BandsTime++)
			{
				// std::cout << "BandsTime: " << BandsTime;

				int ReadSize = (*it)->DimsSize[1] * (*it)->DimsSize[2];
				// short* data_temp = new short[ReadSize];

				(*it)->IsReadData = true;
				unsigned char *DataNew = (unsigned char *)malloc(ReadSize * sizeof(float));

				unsigned char *DataNewDouble = (unsigned char *)malloc(ReadSize * sizeof(double));

				index1[0] = BandsTime;
				index1[1] = 0;
				index1[2] = 0;

				index2[0] = 1;
				index2[1] = (*it)->DimsSize[1];
				index2[2] = (*it)->DimsSize[2];

				mVar_temp.getVar(index1, index2, (short *)DataNew);

				//转换数据排列，上下颠倒

				int RowSize = (*it)->DimsSize[1];
				int ColSize = (*it)->DimsSize[2];
				unsigned char* CurrentRow = NULL;
				unsigned char* AdjustmentMemory = (unsigned char*)malloc(ColSize * sizeof(short));

				unsigned long OneRowSize = (unsigned long)sizeof(short) * (unsigned long)ColSize;
				unsigned long FirstHalfSize = (unsigned long)sizeof(short) * (unsigned long)(ColSize / 2);
				unsigned long SecondHalfSize = OneRowSize - FirstHalfSize;
				for (size_t row = 0; row < RowSize; row++)
				{
					CurrentRow = DataNew + row * OneRowSize;

					memcpy(AdjustmentMemory, CurrentRow, OneRowSize);
					memcpy(CurrentRow, AdjustmentMemory + FirstHalfSize, SecondHalfSize);
					memcpy(CurrentRow + FirstHalfSize, AdjustmentMemory, FirstHalfSize);
				}
				free(AdjustmentMemory);

				for (size_t row = 0; row < RowSize; row++)
				{
					for (size_t col = 0; col < ColSize; col++)
					{
						float temp = ((short*)DataNew)[row * ColSize + col];
						if (temp != (*it)->missing_value)
						{
							if (VarName == "t2m")
							{
								((double*)DataNewDouble)[row * ColSize + col] = temp * (*it)->scale_factor + (*it)->add_offset - 273.15;
							}
							else if (VarName == "ssrd")
							{
								((double*)DataNewDouble)[row * ColSize + col] = (temp * (*it)->scale_factor + (*it)->add_offset) / 3600.0;
							}
							else if (VarName == "d2m")
							{
								((double*)DataNewDouble)[row * ColSize + col] = temp * (*it)->scale_factor + (*it)->add_offset - 273.15;
							}
							else
							{
								((double*)DataNewDouble)[row * ColSize + col] = temp * (*it)->scale_factor + (*it)->add_offset;
							}
						}
						else
						{
							((double*)DataNewDouble)[row * ColSize + col] = temp;
						}
						
					}
				}
				free(DataNew);

				(*it)->Data.push_back(DataNewDouble);
				// std::cout << "共读取：" << index2[1] << "*" << index2[2] << "= " << ReadSize << std::endl;
				// delete[] data_temp;
			}
		}
	}
}

void NetCDF_Read::ReadCollectedData(int BandRead)
{
	for (std::deque<VarInfo *>::iterator it = VarInfoQueue.begin(); it != VarInfoQueue.end(); ++it)
	{
		if ((*it)->DimsCount > 2) //== ReadDataDims)
		{
			ReadData((*it)->Name, BandRead);
		}
	}
}

void NetCDF_Read::TransData(std::deque<VarInfo *> *Destination)
{
	// std::cout << "数据序列长度 " << VarInfoQueue.size() << "准备数据传输... " << std::endl;

	size_t QueueSize = VarInfoQueue.size();

	for (int i = 0; i < QueueSize; i++)
	{
		Destination->push_back(VarInfoQueue.front());
		VarInfoQueue.pop_front();

		// std::cout << VarInfoQueue.size() << " " << Destination->size() << std::endl;
	}
	// std::cout << "数据序列长度 " << Destination->size() << "数据传输完成... " << std::endl;
}

std::string NetCDF_Read::Convert(float Num)
{
	std::ostringstream oss;
	oss << Num;
	std::string str(oss.str());
	return str;
}