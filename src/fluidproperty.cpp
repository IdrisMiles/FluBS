#include "include/fluidproperty.h"

FluidProperty::FluidProperty(unsigned int _numParticles, float _particleMass, float _restDensity, float _smoothingLength, float _deltaTime, unsigned int _solveIterations, unsigned int _gridResolution, float _gridCellWidth):
    numParticles(_numParticles),
    particleMass(_particleMass),
    restDensity(_restDensity),
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
