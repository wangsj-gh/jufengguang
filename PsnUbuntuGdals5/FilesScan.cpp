#include "FilesScan.h"


FilesScan::FilesScan()
{
}


FilesScan::~FilesScan()
{
}


int fileNameFilter(const struct dirent *cur) {
    std::string str(cur->d_name);
    if (str.find(".nc4") != std::string::npos) {
        return 1;
    }
    return 0;
}

void FilesScan::GetFiles(const std::string& dirPath, std::vector<std::string>& files)
{
	struct dirent **namelist;
    // std::vector<std::string> ret;
    std::string p;
    int n = scandir(dirPath.c_str(), &namelist, fileNameFilter, alphasort);
    if (n >= 0) 
    {
        for (int i = 0; i < n; ++i) 
        {
            std::string filePath(namelist[i]->d_name);
            files.push_back(p.assign(dirPath).append("/").append(filePath));
            free(namelist[i]);
        };
        free(namelist);
    }
}


bool FilesScan::ScanFiles(const std::string path)
{
	GetFiles(path, TifFiles);

	std::cout << "TifFiles:" << std::endl;
	for (std::vector<std::string>::iterator it = TifFiles.begin(); it != TifFiles.end(); ++it)
	{
		std::cout << " " << *it << std::endl;
	}

	return true;
}

std::vector<std::string> FilesScan::GetFiles()
{
	std::cout << "共读取文件： " << TifFiles.size() << std::endl;
	return TifFiles;
}

bool FilesScan::Release()
{
	// erase the all elements:
	TifFiles.erase(TifFiles.begin(), TifFiles.end());

	return true;
}