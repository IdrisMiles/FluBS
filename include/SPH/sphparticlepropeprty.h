#ifndef SPHPARTICLEPROPEPRTY_H
#define SPHPARTICLEPROPEPRTY_H

#include <cuda_runtime.h>

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class SphParticleProperty
/// @brief This class holds the properties of the BaseSphParticle
class SphParticleProperty
{

public:

    /// @brief constructor
    SphParticleProperty(unsigned int _numParticles = 8000,
                        float _particleMass = 1.0f,
                        float _particleRadius = 0.2f,
                        float _restDensity = 998.36f,
                        float _smoothingLength = 1.2f,
                        float3 _gravity = make_float3(0.0f, -9.8f, 0.0f)):
            numParticles(_numParticles),
            particleMass(_particleMass),
            particleRadius(_particleRadius),
            restDensity(_restDensity),
            smoothingLength(_smoothingLength),
            gravity(_gravity)
          {

              float dia = 2.0f * particleRadius;
              particleMass = restDensity * (dia * dia * dia);
          }

    /// @brief destructor
    ~SphParticleProperty(){}


    unsigned int numParticles;
    float particleMass;
    float particleRadius;
    float restDensity;
    float smoothingLength;
    float3 gravity;
};

//--------------------------------------------------------------------------------------------------------------

#include "json/src/json.hpp"
using json = nlohmann::json;

void to_json(json& j, const SphParticleProperty& p);
void from_json(const json& j, SphParticleProperty& p);

#endif // SPHPARTICLEPROPEPRTY_H
