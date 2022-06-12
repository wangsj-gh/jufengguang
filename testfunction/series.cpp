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
    // double phi(double x);
    // std::cout << "exp:" << phi(5) << std::endl;
    // for (int  i =0 ; i< 2920;i+=8)
    // {
    //     double h_doy = (double)i / 8 + 1.0;
    //     int h_tiemsers = (int)(i / 8) % 8;
    //     std::cout << "h_doy:" << h_doy<<"  "
    //     "h_tiemsers:"<<h_tiemsers << std::endl;
    // }

    unsigned int aa=1024;
    unsigned int bb=2;
    unsigned int cc=8;
    std::cout <<  aa/bb+cc << std::endl;
    std::cout <<  aa>>1+cc << std::endl;

}

// double phi(double x)
// {
//     return exp(x);
// }