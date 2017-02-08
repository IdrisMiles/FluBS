#include "Fluid/fluidproperty.h"

FluidProperty::FluidProperty(unsigned int _numParticles,
                             float _particleMass,
                             float _particleRadius,
                             float _restDensity,
                             float _surfaceTension,
                             float _surfaceThreshold,
                             float _gasStiffness,
                             float _viscosity,
                             float _smoothingLength,
                             bool _play):
    numParticles(_numParticles),
    particleMass(_particleMass),
    particleRadius(_particleRadius),
    restDensity(_restDensity),
    surfaceTension(_surfaceTension),
    surfaceThreshold(_surfaceThreshold),
    gasStiffness(_gasStiffness),
    viscosity(_viscosity),
    smoothingLength(_smoothingLength),
    play(_play)
{

    float dia = 2.0f * particleRadius;
    particleMass = restDensity * (dia * dia * dia);
}


FluidProperty::~FluidProperty()
{

}
