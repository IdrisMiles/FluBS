#ifndef FLUIDPROPERTY_H
#define FLUIDPROPERTY_H

#include "SPH/sphparticlepropeprty.h"

class FluidProperty : public SphParticleProperty
{

public:
    FluidProperty(unsigned int _numParticles = 8000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _surfaceTension = 0.0728f,
                  float _surfaceThreshold = 1.0f,
                  float _gasStiffness = 100.0f,
                  float _viscosity = 0.1f,
                  float _smoothingLength = 1.2f,
                  bool _play = false):
        SphParticleProperty(_numParticles, _particleMass, _particleRadius, _restDensity, _smoothingLength),
        surfaceTension(_surfaceTension),
        surfaceThreshold(_surfaceThreshold),
        gasStiffness(_gasStiffness),
        viscosity(_viscosity),
        play(_play)
    {

        float dia = 2.0f * particleRadius;
        particleMass = restDensity * (dia * dia * dia);
    }

    ~FluidProperty(){}

    float surfaceTension;
    float surfaceThreshold;
    float gasStiffness;
    float viscosity;

    bool play;
};

#endif // FLUIDPROPERTY_H
