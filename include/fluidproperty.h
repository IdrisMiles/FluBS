#ifndef FLUIDPROPERTY_H
#define FLUIDPROPERTY_H


class FluidProperty
{

public:
    FluidProperty(unsigned int _numParticles = 1000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _surfaceTension = 0.728f,
                  float _surfaceThreshold = 1.0f,//7.065f,
                  float _gasStiffness = 3.5f,
                  float _viscosity = 0.0,//8.9e-4f,
                  float _smoothingLength = 1.0f,
                  float _deltaTime = 0.01f,
                  unsigned int _solveIterations = 1,
                  unsigned int _gridResolution = 10,
                  float _gridCellWidth = 1.2f);

    ~FluidProperty();

    unsigned int numParticles;
    float particleMass;
    float restDensity;
    float surfaceTension;
    float surfaceThreshold;
    float gasStiffness;
    float viscosity;
    float smoothingLength;
    float particleRadius;

    float deltaTime;
    unsigned int solveIterations;
    unsigned int gridResolution;
    float gridCellWidth;
    float gridVolume;
};

#endif // FLUIDPROPERTY_H
