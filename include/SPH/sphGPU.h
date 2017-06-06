#ifndef SPHGPU_H
#define SPHGPU_H

//--------------------------------------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include "SPH/gpudata.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @namespace sphGPU
/// @brief This name space provies a library of functions that link the C++ interface to the relevant CUDA kernels
namespace sphGPU
{

    uint iDivUp(uint a, uint b);

    //--------------------------------------------------------------------------------------------------------------------

    // BaseSPHParticle Reset
    void ResetProperties(ParticleGpuData particle,
                         const uint numCells);

    //--------------------------------------------------------------------------------------------------------------------

    // Algae Reset
    void ResetProperties(AlgaeGpuData particle, const uint numCells);

    //--------------------------------------------------------------------------------------------------------------------

    // Rigid Reset
    void ResetProperties(RigidGpuData particle, const uint numCells);

    //--------------------------------------------------------------------------------------------------------------------

    // Fluid Reset
    void ResetProperties(FluidGpuData particle, const uint numCells);

    //--------------------------------------------------------------------------------------------------------------------

    void ResetTotalForce(float3 *totalForce, const uint numPoints);

    //--------------------------------------------------------------------------------------------------------------------

    void InitFluidAsCube(ParticleGpuData particle, const unsigned int numPartsPerAxis, const float scale);

    //--------------------------------------------------------------------------------------------------------------------

    void InitAlgaeIllumination(float *illum,
                               const unsigned int numPoints);

    //--------------------------------------------------------------------------------------------------------------------

    void InitSphParticleIds(unsigned int *particleId,
                            const unsigned int numPoints);

    //--------------------------------------------------------------------------------------------------------------------

    void ParticleHash(ParticleGpuData particle,
                      const unsigned int gridRes,
                      const float cellWidth);

    //--------------------------------------------------------------------------------------------------------------------

    void SortParticlesByHash(ParticleGpuData particle);

    //--------------------------------------------------------------------------------------------------------------------

    void SortParticlesByHash(AlgaeGpuData particle);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeParticleScatterIds(uint *cellOccupancy,
                                   uint *cellParticleIdx,
                                   const uint numCells);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeMaxCellOccupancy(uint *cellOccupancy,
                                 const uint numCells,
                                 unsigned int &_maxCellOcc);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeParticleVolume(RigidGpuData particle,  const uint gridRes);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeDensity(ParticleGpuData particle, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeDensityFluidFluid(ParticleGpuData particle, ParticleGpuData contributerParticle, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeDensityFluidRigid(ParticleGpuData particle, RigidGpuData rigidParticle, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressureFluid(FluidGpuData particle, const uint gridRes);

    //--------------------------------------------------------------------------------------------------------------------

    void SamplePressure(ParticleGpuData particleData, ParticleGpuData particleContributerData, const uint gridRes);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressureForceFluid(ParticleGpuData particleData, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressureForceFluidFluid(ParticleGpuData particle, ParticleGpuData contributerParticle, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressureForceFluidRigid(ParticleGpuData particle, RigidGpuData rigidParticle, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeViscForce(FluidGpuData particleData, const uint gridRes);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeSurfaceTensionForce(FluidGpuData particleData, const uint gridRes);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeForce(FluidGpuData particleData, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeTotalForce(FluidGpuData particleData,
                           const uint gridRes,
                           const bool accumulatePressure,
                           const bool accumulateViscous,
                           const bool accumulateSurfTen,
                           const bool accumulateExternal,
                           const bool accumulateGravity);

    //--------------------------------------------------------------------------------------------------------------------

    void Integrate(ParticleGpuData particleData, const uint gridRes, const float _dt);

    //--------------------------------------------------------------------------------------------------------------------

    void HandleBoundaries(ParticleGpuData particleData, const uint gridRes, const float _gridDim);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeAdvectionForce(ParticleGpuData particleData, FluidGpuData advectorParticleData, const uint gridRes, const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------
    void AdvectParticle(ParticleGpuData particleData, FluidGpuData advectorParticleData, const uint gridRes, const float deltaTime);

    //--------------------------------------------------------------------------------------------------------------------
    void ComputeBioluminescence(AlgaeGpuData particleData, const uint gridRes, const float deltaTime);

}

//--------------------------------------------------------------------------------------------------------------

#endif // SPHGPU_H
