#ifndef FLUIDPROPERTY_H
#define FLUIDPROPERTY_H


class FluidProperty
{

public:
    FluidProperty(unsigned int _numParticles = 1000/*32768*/, float _particleMass = 0.10f, float _restDensity = 1000.0f, float _smoothingLength = 2.0f, float _deltaTime = 0.16f, unsigned int _solveIterations = 1, unsigned int _gridResolution = 32, float _gridCellWidth = 1.0f);
    ~FluidProperty();

    unsigned int numParticles;
    float particleMass;
    float restDensity;
    float smoothingLength;

    float deltaTime;
    unsigned int solveIterations;
    unsigned int gridResolution;
    float gridCellWidth;
    float gridVolume;
};

#endif // FLUIDPROPERTY_H
