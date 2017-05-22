#include "SPH/rigidproperty.h"


void to_json(json& j, const RigidProperty& p)
{
    j = json();
    j["numPartciles"] = p.numParticles;
    j["particleMass"] = p.particleMass;
    j["particleRadius"] = p.particleRadius;
    j["restDensity"] = p.restDensity;
    j["smoothingLength"] = p.smoothingLength;

    j["m_static"] = p.m_static;
    j["kinematic"] = p.kinematic;
}

void from_json(const json& j, RigidProperty& p)
{
    p.numParticles = j["numPartciles"];
    p.particleMass = j["particleMass"];
    p.particleRadius = j["particleRadius"];
    p.restDensity = j["restDensity"];
    p.smoothingLength = j["smoothingLength"];
    p.m_static = j["m_static"];
    p.kinematic = j["kinematic"];
}
