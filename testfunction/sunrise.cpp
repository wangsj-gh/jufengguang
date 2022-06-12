#include <iostream>
#include <cmath>
using namespace std;

int main(void)
{
    double Lat=40;
    double n=75;
    double delta=23.45*sin((n-80)*3.14/180*360/365)*3.14/180;
    double omegarise=-acos(-tan(Lat*3.14/180)*tan(delta));
    double omegaset=acos(-tan(Lat*3.14/180)*tan(delta));
    double sunrise=12+omegarise*180/3.14/15;
    double sunset=12+omegaset*180/3.14/15;
    // std::cout<<delta<<std::endl;
    std::cout<<sunrise<<std::endl;
    std::cout<<sunset<<std::endl;
  
}

// int main(void)
// {
//     double Lat=40;
//     double n=75;
//     double fai=asin(sin(23.44)*sin(360*n/365));
//     double theta=asin(tan(Lat)*tan(fai));

//     double sunrise=6-theta/360*24;
//     double sunset=6+theta/360*24;

//     // std::cout<<tan(Lat)<<std::endl;
//     std::cout<<fai<<std::endl;
//     std::cout<<theta<<std::endl;
//     std::cout<<sunrise<<std::endl;
//     std::cout<<sunset<<std::endl;
  
// }