#ifndef SPHGPU_KERNELS_H
#define SPHGPU_KERNELS_H


// CUDA includes
#include <cuda_runtime.h>



namespace sphGPU_Kernels
{

    __global__ void ParticleHash_Kernel(uint *hash,
                                        uint *cellOcc,
                                        const float3 *particles,
                                        const uint N,
                                        const uint gridRes,
                                        const float cellWidth);

    __global__ void ComputeVolume_kernel(float *volume,
                                         const uint *cellOcc,
                                         const uint *cellPartIdx,
                                         const float3 *particles,
                                         const uint numPoints,
                                         const float smoothingLength);

    __global__ void ComputeDensity_kernel(float *density,
                                          const float mass,
                                          const uint *cellOcc,
                                          const uint *cellPartIdx,
                                          const float3 *particles,
                                          const uint numPoints,
                                          const float smoothingLength,
                                          const bool accumulate);

    __global__ void ComputeDensityFluidRigid_kernel(const uint numPoints,
                                                    const float fluidRestDensity,
                                                    float *fluidDensity,
                                                    const uint *fluidCellOcc,
                                                    const uint *fluidCellPartIdx,
                                                    const float3 *fluidPos,
                                                    float *rigidVolume,
                                                    const uint *rigidCellOcc,
                                                    const uint *rigidCellPartIdx,
                                                    const float3 *rigidPos,
                                                    const float smoothingLength,
                                                    const bool accumulate);

    __global__ void ComputeDensityFluidFluid_kernel(const uint numPoints,
                                                    float *fluidDensity,
                                                    const uint *fluidCellOcc,
                                                    const uint *fluidCellPartIdx,
                                                    const float3 *fluidPos,
                                                    const uint *otherFluidCellOcc,
                                                    const uint *otherFluidCellPartIdx,
                                                    const float otherFluidMass,
                                                    const float3 *otherFluidPos,
                                                    const float smoothingLength,
                                                    const bool accumulate);

    __global__ void ComputePressure_kernel(float *pressure,
                                           float *density,
                                           const float restDensity,
                                           const float gasConstant,
                                           const uint *cellOcc,
                                           const uint *cellPartIdx,
                                           const uint numPoints);

    __global__ void ComputePressureForce_kernel(float3 *pressureForce,
                                                const float *pressure,
                                                const float *density,
                                                const float mass,
                                                const float3 *particles,
                                                const uint *cellOcc,
                                                const uint *cellPartIdx,
                                                const uint numPoints,
                                                const float smoothingLength,
                                                const bool accumulate);

    __global__ void ComputePressureForceFluidFluid_kernel(float3 *pressureForce,
                                                          const float *pressure,
                                                          const float *density,
                                                          const float mass,
                                                          const float3 *particles,
                                                          const uint *cellOcc,
                                                          const uint *cellPartIdx,
                                                          const float *fluidContribPressure,
                                                          const float *fluidContribDensity,
                                                          const float fluidContribMass,
                                                          const float3 *fluidContribParticles,
                                                          const uint *fluidContribCellOcc,
                                                          const uint *fluidContribCellPartIdx,
                                                          const uint numPoints,
                                                          const float smoothingLength,
                                                          const bool accumulate);

    __global__ void ComputePressureForceFluidRigid_kernel(float3 *pressureForce,
                                                          const float *pressure,
                                                          const float *density,
                                                          const float mass,
                                                          const float3 *particles,
                                                          const uint *cellOcc,
                                                          const uint *cellPartIdx,
                                                          const float restDensity,
                                                          const float *rigidVolume,
                                                          const float3 *rigidPos,
                                                          const uint *rigidCellOcc,
                                                          const uint *rigidCellPartIdx,
                                                          const uint numPoints,
                                                          const float smoothingLength,
                                                          const bool accumulate);

    __global__ void ComputeViscousForce_kernel(float3 *viscForce,
                                               const float viscCoeff,
                                               const float3 *velocity,
                                               const float *density,
                                               const float mass,
                                               const float3 *position,
                                               const uint *cellOcc,
                                               const uint *cellPartIdx,
                                               const uint numPoints,
                                               const float smoothingLength);

    __global__ void ComputeSurfaceTensionForce_kernel(float3 *surfaceTensionForce,
                                                      const float surfaceTension,
                                                      const float surfaceThreshold,
                                                      const float *density,
                                                      const float mass,
                                                      const float3 *position,
                                                      const uint *cellOcc,
                                                      const uint *cellPartIdx,
                                                      const uint numPoints,
                                                      const float smoothingLength);

    __global__ void ComputeForce_kernel(/*float3 *force,*/
                                        float3 *pressureForce,
                                        float3 *viscForce,
                                        float3 *surfaceTensionForce,
//                                        const float3 gravity,
                                        const float viscCoeff,
                                        const float surfaceTension,
                                        const float surfaceThreshold,
                                        const float *pressure,
                                        const float *density,
                                        const float mass,
                                        const float3 *particles,
                                        const float3 *velocity,
                                        const uint *cellOcc,
                                        const uint *cellPartIdx,
                                        const uint numPoints,
                                        const float smoothingLength,
                                        const bool accumulate);

    __global__ void ComputeTotalForce_kernel(const bool accumulatePressure,
                                             const bool accumulateViscous,
                                             const bool accumulateSurfTen,
                                             const bool accumulateExternal,
                                             const bool accumulateGravity,
                                             float3 *force,
                                             const float3 *externalForce,
                                             const float3 *pressureForce,
                                             const float3 *viscousForce,
                                             const float3 *surfaceTensionForce,
                                             const float3 gravity,
                                             const float mass,
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

    //--------------------------------------------------------------------------------------------------------------------
    __global__ void ComputeAdvectionForce(float3 *pos,
                                          float3 *advectForce,
                                          const uint *cellOcc,
                                          const uint *cellPartIdx,
                                          const float3 *advectorPos,
                                          const float3 *advectorForce,
                                          const uint *advectorCellOcc,
                                          const uint *advectorCellPartIdx,
                                          const uint numPoints,
                                          const float smoothingLength,
                                          const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------
    __global__ void AdvectParticle(float3 *pos,
                                   float3 *vel,
                                   const uint *cellOcc,
                                   const uint *cellPartIdx,
                                   const float3 *advectorPos,
                                   const float3 *advectorVel,
                                   const uint *advectorCellOcc,
                                   const uint *advectorCellPartIdx,
                                   const uint numPoints,
                                   const float smoothingLength,
                                   const float deltaTime);

    //--------------------------------------------------------------------------------------------------------------------
    __global__ void ComputeBioluminescence(const float *pressure,
                                           float *prevPressure,
                                           float *illumination,
                                           const uint numPoints);




}

#endif // SPHGPU_KERNELS_H
