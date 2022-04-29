#include <iostream>
#include <deque>
#include <map>
#include <typeinfo>
#include <string.h>

using namespace std;

template <typename T>
void test(T inputData)
{
    if (strcmp(typeid(T).name(), typeid(std::map<std::string, std::deque<double *> *>).name()) == 0)
    {
        std::cout << "the type is map" << std::endl;
    }
    if (strcmp(typeid(T).name(), typeid(std::deque<double *> *).name()) == 0)
    {
        std::cout << "the type is deque" << std::endl;
    }
}

int main(void)
{
    std::map<std::string, std::deque<double *> *> TleafDeque;
    std::deque<double *> *DataDeque;

    test(DataDeque);
}