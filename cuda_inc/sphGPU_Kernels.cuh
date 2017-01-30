#ifndef SPHGPU_KERNELS_H
#define SPHGPU_KERNELS_H


// CUDA includes

#include <cuda_runtime.h>
#include <cuda.h>

#include <math_constants.h>

#include <stdio.h>
#include <math.h>
#include <float.h>



namespace sphGPU_Kernels
{

    __global__ void ParticleHash_Kernel(uint *hash,
                                        uint *cellOcc,
                                        const float3 *particles,
                                        const uint N,
                                        const uint gridRes,
                                        const float cellWidth);

    __global__ void ComputeDensity_kernel(float *density,
                                          const float *mass,
                                          const uint *cellOcc,
                                          const uint *cellPartIdx,
                                          const float3 *particles,
                                          const uint numPoints,
                                          const float smoothingLength);

    __global__ void ComputePressure_kernel(float *pressure,
                                           float *density,
                                           const float restDensity,
                                           const float gasConstant,
                                           const float *mass,
                                           const uint *cellOcc,
                                           const uint *cellPartIdx,
                                           const float3 *particles,
                                           const uint numPoints,
                                           const float smoothingLength);

    __global__ void ComputePressureForce_kernel(float3 *pressureForce,
                                                const float *pressure,
                                                const float *density,
                                                const float *mass,
                                                const float3 *particles,
                                                const uint *cellOcc,
                                                const uint *cellPartIdx,
                                                const uint numPoints,
                                                const float smoothingLength);

    __global__ void ComputeViscousForce_kernel(float3 *viscForce,
                                               const float viscCoeff,
                                               const float3 *velocity,
                                               const float *density,
                                               const float *mass,
                                               const float3 *position,
                                               const uint *cellOcc,
                                               const uint *cellPartIdx,
                                               const uint numPoints,
                                               const float smoothingLength);

    __global__ void ComputeSurfaceTensionForce_kernel(float3 *surfaceTensionForce,
                                                      const float surfaceTension,
                                                      const float surfaceThreshold,
                                                      /*const*/ float *density,
                                                      const float *mass,
                                                      const float3 *position,
                                                      const uint *cellOcc,
                                                      const uint *cellPartIdx,
                                                      const uint numPoints,
                                                      const float smoothingLength);

    __global__ void ComputeForces_kernel(float3 *force,
                                         const float3 *externalForce,
                                         const float3 *pressureForce,
                                         const float3 *viscousForce,
                                         const float3 *surfaceTensionForce,
                                         const float *mass,
                                         const float3 *particles,
                                         const float3 *velocities,
                                         const uint *cellOcc,
                                         const uint *cellPartIdx,
                                         const uint numPoints,
                                         const float smoothingLength);

    __global__ void Integrate_kernel(float3 *force,
                                     float3 *particles,
                                     float3 *velocities,
                                     const float _dt,
                                     const uint numPoints);

    __global__ void HandleBoundaries_Kernel(float3 *particles,
                                            float3 *velocities,
                                            const float boundary,
                                            const uint numPoints);

    __global__ void InitParticleAsCube_Kernel(float3 *particles,
                                              float3 *velocities,
                                              float *densities,
                                              const float restDensity,
                                              const uint numParticles,
                                              const uint numPartsPerAxis,
                                              const float scale);


}

#endif // SPHGPU_KERNELS_H
