#ifndef FLUIDPROPERTY_H
#define FLUIDPROPERTY_H


class FluidProperty
{

public:
    FluidProperty(unsigned int _numParticles = 8000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _surfaceTension = 0.0728f,
                  float _surfaceThreshold = 1.0f,//7.065f,
                  float _gasStiffness = 30.5f,
                  float _viscosity = 8.9e-6f,//4f,
                  float _smoothingLength = 1.2f,
                  float _deltaTime = 0.005f,
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
