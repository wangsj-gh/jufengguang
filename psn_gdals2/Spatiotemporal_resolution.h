#pragma once

#include "StructHeader.h"
using namespace std;

class Spatiotemporal
{
public:
    void ReadSpatiotemporal(int TempResolution, int SpaResolution,std::deque<SpatiotemporalStruct*>* SpatiotemporalDeque);
    void Release();
private:
    int TempResolution;
    int SpaResolution;
    std::deque<SpatiotemporalStruct*>* SpatiotemporalDeque;
};