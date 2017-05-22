#include "SPH/algaeproperty.h"


void to_json(json& j, const AlgaeProperty& p)
{
    j = json();
    j["numPartciles"] = p.numParticles;
    j["particleMass"] = p.particleMass;
    j["particleRadius"] = p.particleRadius;
    j["restDensity"] = p.restDensity;
    j["smoothingLength"] = p.smoothingLength;

    j["bioThreshold"] = p.bioluminescenceThreshold;
    j["reactionRate"] = p.reactionRate;
    j["deactionRate"] = p.deactionRate;
}

void from_json(const json& j, AlgaeProperty& p)
{
    p.numParticles = j["numPartciles"];
    p.particleMass = j["particleMass"];
    p.particleRadius = j["particleRadius"];
    p.restDensity = j["restDensity"];
    p.smoothingLength = j["smoothingLength"];

    p.bioluminescenceThreshold = j["bioThreshold"];
    p.reactionRate = j["reactionRate"];
    p.deactionRate = j["deactionRate"];
}
