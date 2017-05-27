#ifndef GPUDATA_H
#define GPUDATA_H

#include <cuda_runtime.h>

struct ParticleGpuData{
    float3 *pos;
    float3 *vel;
    float *den;
    float *pressure;

    float3* pressureForce;
    float3* gravityForce;
    float3* externalForce;
    float3* totalForce;

    uint *id;
    uint *hash;
    uint *cellOcc;
    uint *cellPartIdx;

    float3 gravity;
    float mass;
    float restDen;
    float radius;
    float smoothingLength;
    uint numParticles;
    uint maxCellOcc;
};

struct FluidGpuData : ParticleGpuData{
    float3 *viscousForce;
    float3 *surfaceTensionForce;

    float surfaceTension;
    float surfaceThreshold;
    float gasStiffness;
    float viscosity;
};

struct AlgaeGpuData : ParticleGpuData
{
    float *prevPressure;
    float *illum;

    float bioluminescenceThreshold;
    float reactionRate;
    float deactionRate;
};

struct RigidGpuData : ParticleGpuData
{
    float *volume;
};


#endif // GPUDATA_H
