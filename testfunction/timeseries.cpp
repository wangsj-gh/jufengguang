#include <iostream>
#include <math.h>
#include <string.h>
using namespace std;

int main(void)
{
    // double timeseries[8]={1.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5};
    int TempResolution = 3;
    int size = 24 / TempResolution;
    double timeSeries[size];
    timeSeries[0] = (double)TempResolution / 2;
    for (int i = 1; i < size; i++)
    {
        timeSeries[i] = timeSeries[i - 1] + (double)TempResolution;
    }

    double timeSeriesNew[size];
    // memcpy(timeSeriesNew, timeSeries, size * sizeof(double));

    for (int j = 0; j < size; j++)
    {
        timeSeriesNew[(j + 1) % size] = timeSeries[j];
    }

    for (int i = 0; i < 25; i++)
    {
        std::cout << ceil(i / 8) + 1 << "  " << timeSeries[i % 8] << " " << timeSeriesNew[i % 8] << std::endl;
    }
}