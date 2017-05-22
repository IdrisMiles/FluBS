#include "SPH/sphparticlepropeprty.h"


void to_json(json& j, const SphParticleProperty& p)
{
    j = json();
    j["numPartciles"] = p.numParticles;
    j["particleMass"] = p.particleMass;
    j["particleRadius"] = p.particleRadius;
    j["restDensity"] = p.restDensity;
    j["smoothingLength"] = p.smoothingLength;
//    j["gravity"] = p.gravity;
}

void from_json(const json& j, SphParticleProperty& p)
{
    p.numParticles = j["numPartciles"];
    p.particleMass = j["particleMass"];
    p.particleRadius = j["particleRadius"];
    p.restDensity = j["restDensity"];
    p.smoothingLength = j["smoothingLength"];
}
