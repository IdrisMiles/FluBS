#ifndef SPHGPU_KERNELS_H
#define SPHGPU_KERNELS_H


// CUDA includes
#include <cuda_runtime.h>
#include "SPH/gpudata.h"

namespace sphGPU_Kernels
{

    __global__ void ParticleHash_Kernel(uint *hash,
                                        uint *cellOcc,
                                        const float3 *particles,
                                        const uint N,
                                        const uint gridRes,
                                        const float cellWidth);

    __global__ void ParticleHash_Kernel(ParticleGpuData particle,
                                        const uint gridRes,
                                        const float cellWidth);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeVolume_kernel(RigidGpuData particle);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeDensity_kernel(ParticleGpuData particle, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeDensityFluidRigid_kernel(ParticleGpuData particle, RigidGpuData rigidParticle, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeDensityFluidFluid_kernel(ParticleGpuData particle, ParticleGpuData contributerParticle, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputePressure_kernel(FluidGpuData particle);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void SamplePressure(ParticleGpuData particleData, ParticleGpuData particleContributerData);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputePressureForce_kernel(ParticleGpuData particleData, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputePressureForceFluidFluid_kernel(ParticleGpuData particle, ParticleGpuData contributerParticle, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputePressureForceFluidRigid_kernel(ParticleGpuData particle, RigidGpuData rigidParticle, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeViscousForce_kernel(FluidGpuData particleData);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeSurfaceTensionForce_kernel(FluidGpuData particleData);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeForce_kernel(FluidGpuData particleData, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void ComputeTotalForce_kernel(const bool accumulatePressure,
                                             const bool accumulateViscous,
                                             const bool accumulateSurfTen,
                                             const bool accumulateExternal,
                                             const bool accumulateGravity,
                                             FluidGpuData particleData);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void Integrate_kernel(ParticleGpuData particleData, const float _dt);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void HandleBoundaries_Kernel(ParticleGpuData particleData, const float boundary);

    //--------------------------------------------------------------------------------------------------------------------

    __global__ void InitParticleAsCube_Kernel(ParticleGpuData particle,
                                              const uint numPartsPerAxis,
                                              const float scale);

    //--------------------------------------------------------------------------------------------------------------------
    __global__ void ComputeAdvectionForce(ParticleGpuData particleData, FluidGpuData advectorParticleData, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------
    __global__ void AdvectParticle(ParticleGpuData particleData, FluidGpuData advectorParticleData, const float deltaTime);

    //--------------------------------------------------------------------------------------------------------------------
    __global__ void ComputeBioluminescence(AlgaeGpuData particleData, const float deltaTime);




}

#endif // SPHGPU_KERNELS_H
