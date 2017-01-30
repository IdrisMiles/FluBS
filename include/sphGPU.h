#ifndef SPHGPU_H
#define SPHGPU_H

#include <cuda_runtime.h>
#include <cuda.h>


namespace sphGPU
{

    uint iDivUp(uint a, uint b);

    void InitFluidAsCube(float3 *particles,
                         float3 *velocities,
                         float *densities,
                         const float restDensity,
                         const unsigned int numParticles,
                         const unsigned int numPartsPerAxis,
                         const float scale);

    void ParticleHash(unsigned int *hash,
                      unsigned int *cellOcc,
                      float3 *particles,
                      const unsigned int numPoints,
                      const unsigned int gridRes,
                      const float cellWidth);

    void ComputePressure(const uint maxCellOcc, const uint gridRes,
                         float *pressure,
                         float *density,
                         const float restDensity,
                         const float gasConstant,
                         const float *mass,
                         const uint *cellOcc,
                         const uint *cellPartIdx,
                         const float3 *particles,
                         const uint numPoints,
                         const float smoothingLength);

    void ComputePressureForce(const uint maxCellOcc,
                              const uint gridRes,
                              float3 *pressureForce,
                              const float *pressure,
                              const float *density,
                              const float *mass,
                              const float3 *particles,
                              const uint *cellOcc,
                              const uint *cellPartIdx,
                              const uint numPoints,
                              const float smoothingLength);

    void ComputeViscForce(const uint maxCellOcc,
                          const uint gridRes,
                          float3 *viscForce,
                          const float viscCoeff,
                          const float3 *velocity,
                          const float *density,
                          const float *mass,
                          const float3 *particles,
                          const uint *cellOcc,
                          const uint *cellPartIdx,
                          const uint numPoints,
                          const float smoothingLength);

    void ComputeSurfaceTensionForce(const uint maxCellOcc,
                                    const uint gridRes,
                                    float3 *surfTenForce,
                                    const float surfaceTension,
                                    const float surfaceThreshold,
                                    float *density,
                                    const float *mass,
                                    const float3 *particles,
                                    const uint *cellOcc,
                                    const uint *cellPartIdx,
                                    const uint numPoints,
                                    const float smoothingLength);

    void ComputeTotalForce(const uint maxCellOcc,
                           const uint gridRes,
                           float3 *force,
                           const float3 *externalForce,
                           const float3 *pressureForce,
                           const float3 *viscForce,
                           const float3 *surfaceTensionForce,
                           const float *mass,
                           const float3 *particles,
                           const float3 *velocities,
                           const uint *cellOcc,
                           const uint *cellPartIdx,
                           const uint numPoints,
                           const float smoothingLength);

    void Integrate(const uint maxCellOcc,
                   const uint gridRes,
                   float3 *force,
                   float3 *particles,
                   float3 *velocities,
                   const float _dt,
                   const uint numPoints);

    void HandleBoundaries(const uint maxCellOcc,
                          const uint gridRes,
                          float3 *particles,
                          float3 *velocities,
                          const float _gridDim,
                          const uint numPoints);
}

#endif // SPHGPU_H
