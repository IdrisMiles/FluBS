#include "include/fluidproperty.h"

FluidProperty::FluidProperty(unsigned int _numParticles,
                             float _particleMass,
                             float _particleRadius,
                             float _restDensity,
                             float _surfaceTension,
                             float _surfaceThreshold,
                             float _gasStiffness,
                             float _viscosity,
                             float _smoothingLength,
                             float _deltaTime,
                             unsigned int _solveIterations,
                             unsigned int _gridResolution,
                             float _gridCellWidth):
    numParticles(_numParticles),
    particleMass(_particleMass),
    particleRadius(_particleRadius),
    restDensity(_restDensity),
    surfaceTension(_surfaceTension),
    surfaceThreshold(_surfaceThreshold),
    gasStiffness(_gasStiffness),
    viscosity(_viscosity),
    smoothingLength(_smoothingLength),
    deltaTime(_deltaTime),
    solveIterations(_solveIterations),
    gridResolution(_gridResolution),
    gridCellWidth(_gridCellWidth)
{
}


FluidProperty::~FluidProperty()
{

}
