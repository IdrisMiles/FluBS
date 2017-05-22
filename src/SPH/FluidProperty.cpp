#include "SPH/fluidproperty.h"


void to_json(json& j, const FluidProperty& p)
{
    j = json();
    j["numPartciles"] = p.numParticles;
    j["particleMass"] = p.particleMass;
    j["particleRadius"] = p.particleRadius;
    j["restDensity"] = p.restDensity;
    j["smoothingLength"] = p.smoothingLength;

    j["surfaceTension"] = p.surfaceTension;
    j["surfaceThreshold"] = p.surfaceThreshold;
    j["gasStiffness"] = p.gasStiffness;
    j["viscosity"] = p.viscosity;
}

void from_json(const json& j, FluidProperty& p)
{
    p.numParticles = j["numPartciles"];
    p.particleMass = j["particleMass"];
    p.particleRadius = j["particleRadius"];
    p.restDensity = j["restDensity"];
    p.smoothingLength = j["smoothingLength"];

    p.surfaceTension = j["surfaceTension"];
    p.surfaceThreshold = j["surfaceThreshold"];
    p.gasStiffness = j["gasStiffness"];
    p.viscosity = j["viscosity"];
}
