#ifndef ALGAEPROPERTIES_H
#define ALGAEPROPERTIES_H


#include "SPH/sphparticlepropeprty.h"

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class FluidProperty
/// @brief This class holds the properties of the Fluid sph particle
class AlgaeProperty : public SphParticleProperty
{

public:
    /// @brief constructor
    AlgaeProperty(float _bioluminescenceThreshold = 200.0f,
                  float _reactionRate = 1.0f,
                  float _deactionRate = 1.0f,
                  unsigned int _numParticles = 16000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _smoothingLength = 1.2f,
                  float3 _gravity = make_float3(0.0f, -9.8f, 0.0f)):
        SphParticleProperty(_numParticles, _particleMass, _particleRadius, _restDensity, _smoothingLength, _gravity),
        bioluminescenceThreshold(_bioluminescenceThreshold),
        reactionRate(_reactionRate),
        deactionRate(_deactionRate)
    {
        float dia = 2.0f * particleRadius;
        particleMass = restDensity * (dia * dia * dia);
    }

    /// @brief destructor
    ~AlgaeProperty(){}

    //--------------------------------------------------------------------------------------------------------------

    float bioluminescenceThreshold;
    float reactionRate;
    float deactionRate;
};

//--------------------------------------------------------------------------------------------------------------

void to_json(json& j, const AlgaeProperty& p);
void from_json(const json& j, AlgaeProperty& p);

//--------------------------------------------------------------------------------------------------------------

#endif // ALGAEPROPERTIES_H
