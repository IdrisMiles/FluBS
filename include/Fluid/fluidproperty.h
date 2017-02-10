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
                  float _surfaceThreshold = 1.0f, //7.065f,
                  float _gasStiffness = 30.5f,
                  float _viscosity = 1.0e-2f,
                  float _smoothingLength = 1.2f,
                  bool _play = false):
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

    ~FluidProperty(){}

    unsigned int numParticles;
    float particleMass;
    float restDensity;
    float surfaceTension;
    float surfaceThreshold;
    float gasStiffness;
    float viscosity;
    float particleRadius;
    float smoothingLength;

    bool play;
};

#endif // FLUIDPROPERTY_H
