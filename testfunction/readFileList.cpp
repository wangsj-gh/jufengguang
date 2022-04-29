#include <cstring>
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
std::vector<std::string> readFileList(const char *basePath)
{
    std::vector<std::string> result;
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
            // {printf("d_name:%s/%s\n",basePath,ptr->d_name);
            {result.push_back(std::string(ptr->d_name));}
        else if(ptr->d_type == 10)    ///link file
            // {printf("d_name:%s/%s\n",basePath,ptr->d_name);
            {result.push_back(std::string(ptr->d_name));}
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            result.push_back(std::string(ptr->d_name));
            readFileList(base);
        }
    }
    closedir(dir);
    return result;
}

int main(void)
{
    const char *filePath="/data/users/wangsj/dataGdals/lai2007";
    std::vector<std::string> FileList;
    FileList=readFileList(filePath);
    for (vector<std::string>::iterator it = FileList.begin(); it != FileList.end(); it++) {
        cout << *it << " "<<endl;
	}
    // std::cout<<FileList.size()<<std::endl;
    return 0;
}