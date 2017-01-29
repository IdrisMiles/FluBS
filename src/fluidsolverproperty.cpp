#include "include/fluidsolverproperty.h"

FluidSolverProperty::FluidSolverProperty(float _smoothingLength,
                                         float _deltaTime,
                                         unsigned int _solveIterations,
                                         unsigned int _gridResolution,
                                         float _gridCellWidth,
                                         bool _play):
    smoothingLength(_smoothingLength),
    deltaTime(_deltaTime),
    solveIterations(_solveIterations),
    gridResolution(_gridResolution),
    gridCellWidth(_gridCellWidth),
    play(_play)
{

}


FluidSolverProperty::~FluidSolverProperty()
{

}
