#ifndef RIGIDPROPERTY_H
#define RIGIDPROPERTY_H

#include "SPH/sphparticlepropeprty.h"

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------

/// @class RigidProperty
/// @brief This class holds the properties of the Rigid sph particle
class RigidProperty : public SphParticleProperty
{

public:
    /// @brief constructor
    RigidProperty(int _static = true,
                  int _kinematic = false,
                  unsigned int _numParticles = 8000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _smoothingLength = 1.2f,
                  float3 _gravity = make_float3(0.0f, -9.8f, 0.0f)):
        SphParticleProperty(_numParticles, _particleMass, _particleRadius, _restDensity, _smoothingLength, _gravity),
        m_static(_static),
        kinematic(_kinematic)
    {

        float dia = 2.0f * particleRadius;
        particleMass = restDensity * (dia * dia * dia);
    }

    /// @brief destructor
    ~RigidProperty(){}

    //--------------------------------------------------------------------------------------------------------------

    int m_static;
    int kinematic;
};

//--------------------------------------------------------------------------------------------------------------

void to_json(json& j, const RigidProperty& p);
void from_json(const json& j, RigidProperty& p);

#endif // RIGIDPROPERTY_H
