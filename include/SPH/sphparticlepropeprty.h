#ifndef SPHPARTICLEPROPEPRTY_H
#define SPHPARTICLEPROPEPRTY_H

class SphParticleProperty
{

public:

    SphParticleProperty(unsigned int _numParticles = 8000,
                        float _particleMass = 1.0f,
                        float _particleRadius = 0.2f,
                        float _restDensity = 998.36f,
                        float _smoothingLength = 1.2f):
              numParticles(_numParticles),
              particleMass(_particleMass),
              particleRadius(_particleRadius),
              restDensity(_restDensity),
              smoothingLength(_smoothingLength)
          {

              float dia = 2.0f * particleRadius;
              particleMass = restDensity * (dia * dia * dia);
          }

    ~SphParticleProperty(){}

    unsigned int numParticles;
    float particleMass;
    float particleRadius;
    float restDensity;
    float smoothingLength;
};

#endif // SPHPARTICLEPROPEPRTY_H
