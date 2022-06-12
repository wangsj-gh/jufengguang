#include "StructHeader.h"
#include "Spatiotemporal_resolution.h"

void Spatiotemporal::ReadSpatiotemporal(int TempResolution, int SpaResolution, std::deque<SpatiotemporalStruct *> *SpatiotemporalDeque)
{

    SpatiotemporalDeque->push_back(new SpatiotemporalStruct);
    SpatiotemporalDeque->back()->TempResolution = TempResolution;
    SpatiotemporalDeque->back()->SpaResolution = SpaResolution;
    SpatiotemporalDeque->back()->TimeSize = 24 / TempResolution;
    SpatiotemporalDeque->back()->timeSeries = new double[SpatiotemporalDeque->back()->TimeSize];

    double timeSeriesold[SpatiotemporalDeque->back()->TimeSize];
    timeSeriesold[0] = (double)TempResolution / 2;
    for (int i = 1; i < SpatiotemporalDeque->back()->TimeSize; i++)
    {
        timeSeriesold[i] = timeSeriesold[i - 1] + (double)TempResolution;
    }

    for (int j = 0; j < SpatiotemporalDeque->back()->TimeSize; j++)
    {
        SpatiotemporalDeque->back()->timeSeries[(j + 1) % SpatiotemporalDeque->back()->TimeSize] = timeSeriesold[j];
    }
}

void Spatiotemporal::Release()
{
    for (std::deque<SpatiotemporalStruct *>::iterator it = SpatiotemporalDeque->begin(); it != SpatiotemporalDeque->end(); it++)
    {
        delete[](*it)->timeSeries;
    }
}