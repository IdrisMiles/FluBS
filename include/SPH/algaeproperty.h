#ifndef ALGAEPROPERTIES_H
#define ALGAEPROPERTIES_H


#include "SPH/sphparticlepropeprty.h"

class AlgaeProperty : public SphParticleProperty
{

public:
    AlgaeProperty(unsigned int _numParticles = 16000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _smoothingLength = 1.2f,
                  float3 _gravity = make_float3(0.0f, -9.8f, 0.0f)):
        SphParticleProperty(_numParticles, _particleMass, _particleRadius, _restDensity, _smoothingLength, _gravity)
    {

        float dia = 2.0f * particleRadius;
        particleMass = restDensity * (dia * dia * dia);
    }

    ~AlgaeProperty(){}
};


#endif // ALGAEPROPERTIES_H
