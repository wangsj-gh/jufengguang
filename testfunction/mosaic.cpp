// #include <iterator>
#include <iostream>
#include <vector>
#include <string>
// #include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <thread>

using namespace std;

int fileNameFilter(const struct dirent *cur)
{
    std::string str(cur->d_name);
    if (str.find(".tif") != std::string::npos)
    {
        return 1;
    }
    return 0;
}

std::vector<std::string> getDirBinsSortedPath(std::string dirPath)
{
    struct dirent **namelist;
    std::vector<std::string> ret;
    std::string p;
    int n = scandir(dirPath.c_str(), &namelist, fileNameFilter, alphasort);
    if (n < 0)
    {
        return ret;
    }
    for (int i = 0; i < n; ++i)
    {
        std::string filePath(namelist[i]->d_name);
        ret.push_back(p.assign(dirPath).append("/").append(filePath));
        free(namelist[i]);
    };
    free(namelist);
    return ret;
}

int main(void)
{
    std::string filePath = "/data/appdata/lai_param_TwoCycle_mask/2000";
    std::vector<std::string> FileList;
    FileList = getDirBinsSortedPath(filePath);
    for (vector<std::string>::iterator it = FileList.begin(); it != FileList.end(); it++)
    {
        // sleep(0.1);
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        cout << *it << " " << endl;
    }
    std::cout << "total size:" << FileList.size() << std::endl;
    return 0;
}
