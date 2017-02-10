#ifndef RIGIDPROPERTY_H
#define RIGIDPROPERTY_H


class RigidProperty
{

public:
    RigidProperty(unsigned int _numParticles = 8000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _smoothingLength = 1.2f,
                  bool _kinematic = false):
        numParticles(_numParticles),
        particleMass(_particleMass),
        particleRadius(_particleRadius),
        restDensity(_restDensity),
        smoothingLength(_smoothingLength),
        m_kinematic(_kinematic)
    {

        float dia = 2.0f * particleRadius;
        particleMass = restDensity * (dia * dia * dia);
    }

    ~RigidProperty(){}

    unsigned int numParticles;
    float particleMass;
    float particleRadius;
    float restDensity;
    float smoothingLength;
    bool m_kinematic;
};


#endif // RIGIDPROPERTY_H
