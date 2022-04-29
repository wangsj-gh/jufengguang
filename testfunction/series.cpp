#include <iostream>
#include <cmath>
#include "string.h"
using namespace std;
int main(void)
{
    // int numElements = 1440 * 600;
    // size_t size = numElements * sizeof(double);
    // double h_annsum_mr[numElements];
    // memset(h_annsum_mr, 0, size);
    // for (int i = 0; i < numElements; i++)
    // {
    //     std::cout << i << "=" << h_annsum_mr[i] << std::endl;
    // }
    double phi(double x);
    std::cout << "exp:" << phi(5) << std::endl;
}

double phi(double x)
{
    return exp(x);
}