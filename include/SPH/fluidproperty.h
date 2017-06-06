#ifndef FLUIDPROPERTY_H
#define FLUIDPROPERTY_H

#include "SPH/sphparticlepropeprty.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------

/// @class FluidProperty
/// @brief This class holds the properties of the Fluid sph particle
class FluidProperty : public SphParticleProperty
{

public:
    /// @brief constructor
    FluidProperty(float _surfaceTension = 0.0728f,
                  float _surfaceThreshold = 1.0f,
                  float _gasStiffness = 50.0f,
                  float _viscosity = 0.1f,
                  unsigned int _numParticles = 16000,
                  float _particleMass = 1.0f,
                  float _particleRadius = 0.2f,
                  float _restDensity = 998.36f,
                  float _smoothingLength = 1.2f,
                  float3 _gravity = make_float3(0.0f, -9.8f, 0.0f)):
        SphParticleProperty(_numParticles, _particleMass, _particleRadius, _restDensity, _smoothingLength, _gravity),
        surfaceTension(_surfaceTension),
        surfaceThreshold(_surfaceThreshold),
        gasStiffness(_gasStiffness),
        viscosity(_viscosity)
    {
        float dia = 2.0f * particleRadius;
        particleMass = restDensity * (dia * dia * dia);
    }

    /// @brief destructor
    ~FluidProperty(){}


    //--------------------------------------------------------------------------------------------------------------

    float surfaceTension;
    float surfaceThreshold;
    float gasStiffness;
    float viscosity;

};

//--------------------------------------------------------------------------------------------------------------

void to_json(json& j, const FluidProperty& p);
void from_json(const json& j, FluidProperty& p);

//--------------------------------------------------------------------------------------------------------------
#endif // FLUIDPROPERTY_H
