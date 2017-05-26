#ifndef SPHGPU_H
#define SPHGPU_H

#include <cuda_runtime.h>
#include "SPH/gpudata.h"

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

    void ComputePressureForceFluid(const uint maxCellOcc,
                              const uint gridRes,
                              float3 *pressureForce,
                              const float *pressure,
                              const float *density,
                              const float mass,
                              const float3 *particles,
                              const uint *cellOcc,
                              const uint *cellPartIdx,
                              const uint numPoints,
                              const float smoothingLength,
                              const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressureForceFluidFluid(const uint maxCellOcc,
                                        const uint gridRes,
                                        float3 *pressureForce,
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

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressureForceFluidRigid(const uint maxCellOcc,
                                        const uint gridRes,
                                        float3 *pressureForce,
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

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeViscForce(const uint maxCellOcc,
                          const uint gridRes,
                          float3 *viscForce,
                          const float viscCoeff,
                          const float3 *velocity,
                          const float *density,
                          const float mass,
                          const float3 *particles,
                          const uint *cellOcc,
                          const uint *cellPartIdx,
                          const uint numPoints,
                          const float smoothingLength);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeSurfaceTensionForce(const uint maxCellOcc,
                                    const uint gridRes,
                                    float3 *surfTenForce,
                                    const float surfaceTension,
                                    const float surfaceThreshold,
                                    const float *density,
                                    const float mass,
                                    const float3 *particles,
                                    const uint *cellOcc,
                                    const uint *cellPartIdx,
                                    const uint numPoints,
                                    const float smoothingLength);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeForce(const uint maxCellOcc,
                        const uint gridRes,
                        float3 *pressureForce,
                        float3 *viscForce,
                        float3 *surfaceTensionForce,
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

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeTotalForce(const uint maxCellOcc,
                           const uint gridRes,
                           const bool accumulatePressure,
                           const bool accumulateViscous,
                           const bool accumulateSurfTen,
                           const bool accumulateExternal,
                           const bool accumulateGravity,
                           float3 *force,
                           const float3 *externalForce,
                           const float3 *pressureForce,
                           const float3 *viscForce,
                           const float3 *surfaceTensionForce,
                           const float3 gravity,
                           const float mass,
                           const float3 *particles,
                           const float3 *velocities,
                           const uint *cellOcc,
                           const uint *cellPartIdx,
                           const uint numPoints,
                           const float smoothingLength);

    //--------------------------------------------------------------------------------------------------------------------

    void Integrate(const uint maxCellOcc,
                   const uint gridRes,
                   float3 *force,
                   float3 *particles,
                   float3 *velocities,
                   const float _dt,
                   const uint numPoints);

    //--------------------------------------------------------------------------------------------------------------------

    void HandleBoundaries(const uint maxCellOcc,
                          const uint gridRes,
                          float3 *particles,
                          float3 *velocities,
                          const float _gridDim,
                          const uint numPoints);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeAdvectionForce(const uint maxCellOcc,
                               const uint gridRes,
                               float3 *pos,
                               float3 *vel,
                               float3 *advectForce,
                               const uint *cellOcc,
                               const uint *cellPartIdx,
                               const float3 *advectorPos,
                               const float3 *advectorForce,
                               const float *advectorDensity,
                               const float advectorMass,
                               const uint *advectorCellOcc,
                               const uint *advectorCellPartIdx,
                               const uint numPoints,
                               const float smoothingLength,
                               const bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------
    void AdvectParticle(const uint maxCellOcc,
                        const uint gridRes,
                        float3 *pos,
                        float3 *vel,
                        const uint *cellOcc,
                        const uint *cellPartIdx,
                        const float3 *advectorPos,
                        const float3 *advectorVel,
                        const float *advectorDensity,
                        const float advectorMass,
                        const uint *advectorCellOcc,
                        const uint *advectorCellPartIdx,
                        const uint numPoints,
                        const float smoothingLength,
                        const float deltaTime);

    //--------------------------------------------------------------------------------------------------------------------
    void ComputeBioluminescence(const uint maxCellOcc,
                                const uint gridRes,
                                const float *pressure,
                                float *prevPressure,
                                float *illumination,
                                const float bioThreshold,
                                const float reactionRate,
                                const float deactionRate,
                                const float deltaTime,
                                const uint numPoints);


    //--------------------------------------------------------------------------------------------------------------------
    namespace pci
    {
//        void PredictIntegrate(const uint maxCellOcc,
//                              const uint gridRes,
//                              float3 *force,
//                              float3 *particles,
//                              float3 *velocities,
//                              const float _dt,
//                              const uint numPoints);

//        void PredictDensity(const uint maxCellOcc,
//                   const uint gridRes,);

        void predictDensityVariation(const uint maxCellOcc,
                                     const uint gridRes);

        void ComputeMaxDensityVariation(const uint maxCellOcc,
                                        const uint gridRes,
                                        float &_maxDenVar);

        void UpdatePressure(const uint maxCellOcc,
                            const uint gridRes);

        void ComputePressureForce(const uint maxCellOcc,
                                  const uint gridRes);


    }
}

#endif // SPHGPU_H
