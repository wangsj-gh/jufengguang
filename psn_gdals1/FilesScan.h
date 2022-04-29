#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>

class FilesScan
{
public:
	FilesScan();
	~FilesScan();

    // int fileNameFilter(const struct dirent *cur) ;
	void GetFiles(const std::string& path, std::vector<std::string>& files);
	bool ScanFiles(const std::string path);

	std::vector<std::string> GetFiles();


	bool Release();

private:

	std::vector<std::string> TifFiles;
};

