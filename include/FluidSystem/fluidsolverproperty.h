#ifndef FLUIDSOLVERPROPERTY_H
#define FLUIDSOLVERPROPERTY_H

#include "json/src/json.hpp"
using json = nlohmann::json;


class FluidSolverProperty
{
public:
    FluidSolverProperty(float _smoothingLength = 1.2f,
                        float _deltaTime = 0.005f,
                        unsigned int _solveIterations = 1,
                        unsigned int _gridResolution = 20,
                        float _gridCellWidth = 1.2f):
        smoothingLength(_smoothingLength),
        deltaTime(_deltaTime),
        solveIterations(_solveIterations),
        gridResolution(_gridResolution),
        gridCellWidth(_gridCellWidth)
    {

    }

    FluidSolverProperty(const FluidSolverProperty &_copy)
    {
        this->smoothingLength = _copy.smoothingLength;
        this->deltaTime = _copy.deltaTime;
        this->solveIterations = _copy.solveIterations;
        this->gridResolution = _copy.gridResolution;
        this->gridCellWidth = _copy.gridCellWidth;
    }

    FluidSolverProperty operator= (const FluidSolverProperty &_rhs)
    {
        this->smoothingLength = _rhs.smoothingLength;
        this->deltaTime = _rhs.deltaTime;
        this->solveIterations = _rhs.solveIterations;
        this->gridResolution = _rhs.gridResolution;
        this->gridCellWidth = _rhs.gridCellWidth;
    }

    ~FluidSolverProperty(){}


    float smoothingLength;
    float deltaTime;
    unsigned int solveIterations;
    unsigned int gridResolution;
    float gridCellWidth;
    float gridVolume;

};

void to_json(json& j, const FluidSolverProperty& p);
void from_json(const json& j, FluidSolverProperty& p);

#endif // FLUIDSOLVERPROPERTY_H
