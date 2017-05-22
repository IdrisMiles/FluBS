#include "FluidSystem/fluidsolverproperty.h"


void to_json(json& j, const FluidSolverProperty& p)
{
    j = json();
    j["smoothingLength"] = p.smoothingLength;
    j["deltaTime"] = p.deltaTime;
    j["solveIterations"] = p.solveIterations;
    j["gridResolution"] = p.gridResolution;
    j["gridCellWidth"] = p.gridCellWidth;
    j["gridVolume"] = p.gridVolume;
}

void from_json(const json& j, FluidSolverProperty& p)
{
    p.smoothingLength = j["smoothingLength"];
    p.deltaTime = j["deltaTime"];
    p.solveIterations = j["solveIterations"];
    p.gridResolution = j["gridResolution"];
    p.gridCellWidth = j["gridCellWidth"];
    p.gridVolume = j["gridVolume"];
}
