#ifndef FLUIDPROPERTY_H
#define FLUIDPROPERTY_H


class FluidProperty
{

public:
    FluidProperty(unsigned int _numParticles = 32768, float _particleMass = 1.0f, float _particleRadius = 0.1f, float _restDensity = 1000.0f, float _smoothingLength = 1.0f, float _deltaTime = 0.01f, unsigned int _solveIterations = 1, unsigned int _gridResolution = 16, float _gridCellWidth = 1.0f);
    ~FluidProperty();

    unsigned int numParticles;
    float particleMass;
    float restDensity;
    float smoothingLength;
    float particleRadius;

    float deltaTime;
    unsigned int solveIterations;
    unsigned int gridResolution;
    float gridCellWidth;
    float gridVolume;
};

#endif // FLUIDPROPERTY_H
