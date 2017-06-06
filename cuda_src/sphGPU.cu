#include "../include/SPH/sphGPU.h"
#include "../cuda_inc/sphGPU_Kernels.cuh"


#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include <algorithm>

//--------------------------------------------------------------------------------------------------------------------

uint sphGPU::iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ResetProperties(ParticleGpuData particle, const uint numCells)
{
    thrust::device_ptr<float3> pressureForcePtr = thrust::device_pointer_cast(particle.pressureForce);
    thrust::device_ptr<float3> externalForcePtr = thrust::device_pointer_cast(particle.externalForce);
    thrust::device_ptr<float3> totalForcePtr = thrust::device_pointer_cast(particle.totalForce);
    thrust::device_ptr<float> densityPtr = thrust::device_pointer_cast(particle.den);
    thrust::device_ptr<float> pressurePtr = thrust::device_pointer_cast(particle.pressure);
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(particle.hash);
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(particle.cellOcc);
    thrust::device_ptr<uint> cellPartIdxPtr = thrust::device_pointer_cast(particle.cellPartIdx);

    thrust::fill(pressureForcePtr, pressureForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(externalForcePtr, externalForcePtr+particle.numParticles, make_float3(0.0f, 0.0f,0.0f));
    thrust::fill(totalForcePtr, totalForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(densityPtr, densityPtr+particle.numParticles, 0.0f);
    thrust::fill(pressurePtr, pressurePtr+particle.numParticles, 0.0f);
    thrust::fill(hashPtr, hashPtr+particle.numParticles, 0);
    thrust::fill(cellOccPtr, cellOccPtr+numCells, 0);
    thrust::fill(cellPartIdxPtr, cellPartIdxPtr+numCells, 0);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ResetProperties(AlgaeGpuData particle, const uint numCells)
{
    thrust::device_ptr<float3> pressureForcePtr = thrust::device_pointer_cast(particle.pressureForce);
    thrust::device_ptr<float3> externalForcePtr = thrust::device_pointer_cast(particle.externalForce);
    thrust::device_ptr<float3> totalForcePtr = thrust::device_pointer_cast(particle.totalForce);
    thrust::device_ptr<float> densityPtr = thrust::device_pointer_cast(particle.den);
    thrust::device_ptr<float> pressurePtr = thrust::device_pointer_cast(particle.pressure);
//    thrust::device_ptr<float> prevPressurePtr = thrust::device_pointer_cast(particle.prevPressure);
//    thrust::device_ptr<float> bioIllumPtr = thrust::device_pointer_cast(particle.bioIllum);
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(particle.hash);
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(particle.cellOcc);
    thrust::device_ptr<uint> cellPartIdxPtr = thrust::device_pointer_cast(particle.cellPartIdx);

    thrust::fill(pressureForcePtr, pressureForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(externalForcePtr, externalForcePtr+particle.numParticles, make_float3(0.0f, 0.0f,0.0f));
    thrust::fill(totalForcePtr, totalForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(densityPtr, densityPtr+particle.numParticles, 0.0f);
    thrust::fill(pressurePtr, pressurePtr+particle.numParticles, 0.0f);
//    thrust::fill(prevPressurePtr, prevPressurePtr+particle.numParticles, 0.0f);
//    thrust::fill(bioIllumPtr, bioIllumPtr+particle.numParticles, 0.0f);
    thrust::fill(hashPtr, hashPtr+particle.numParticles, 0);

    thrust::fill(cellOccPtr, cellOccPtr+numCells, 0);
    thrust::fill(cellPartIdxPtr, cellPartIdxPtr+numCells, 0);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ResetProperties(RigidGpuData particle, const uint numCells)
{
    thrust::device_ptr<float3> pressureForcePtr = thrust::device_pointer_cast(particle.pressureForce);
    thrust::device_ptr<float3> externalForcePtr = thrust::device_pointer_cast(particle.externalForce);
    thrust::device_ptr<float3> totalForcePtr = thrust::device_pointer_cast(particle.totalForce);
    thrust::device_ptr<float> densityPtr = thrust::device_pointer_cast(particle.den);
    thrust::device_ptr<float> pressurePtr = thrust::device_pointer_cast(particle.pressure);
    thrust::device_ptr<float> volumePtr = thrust::device_pointer_cast(particle.volume);
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(particle.hash);
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(particle.cellOcc);
    thrust::device_ptr<uint> cellPartIdxPtr = thrust::device_pointer_cast(particle.cellPartIdx);

    thrust::fill(pressureForcePtr, pressureForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(externalForcePtr, externalForcePtr+particle.numParticles, make_float3(0.0f, 0.0f,0.0f));
    thrust::fill(totalForcePtr, totalForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(densityPtr, densityPtr+particle.numParticles, 0.0f);
    thrust::fill(pressurePtr, pressurePtr+particle.numParticles, 0.0f);
    thrust::fill(volumePtr, volumePtr+particle.numParticles, 1.0f);
    thrust::fill(hashPtr, hashPtr+particle.numParticles, 0);
    thrust::fill(cellOccPtr, cellOccPtr+numCells, 0);
    thrust::fill(cellPartIdxPtr, cellPartIdxPtr+numCells, 0);
}

//--------------------------------------------------------------------------------------------------------------------


void sphGPU::ResetProperties(FluidGpuData particle, const uint numCells)
{
    thrust::device_ptr<float3> pressureForcePtr = thrust::device_pointer_cast(particle.pressureForce);
    thrust::device_ptr<float3> viscousForcePtr = thrust::device_pointer_cast(particle.viscousForce);
    thrust::device_ptr<float3> surfTenForcePtr = thrust::device_pointer_cast(particle.surfaceTensionForce);
    thrust::device_ptr<float3> externalForcePtr = thrust::device_pointer_cast(particle.externalForce);
    thrust::device_ptr<float3> totalForcePtr = thrust::device_pointer_cast(particle.totalForce);
//    thrust::device_ptr<float> densityErrPtr = thrust::device_pointer_cast(particle.densityErr);
    thrust::device_ptr<float> densityPtr = thrust::device_pointer_cast(particle.den);
    thrust::device_ptr<float> pressurePtr = thrust::device_pointer_cast(particle.pressure);
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(particle.hash);
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(particle.cellOcc);
    thrust::device_ptr<uint> cellPartIdxPtr = thrust::device_pointer_cast(particle.cellPartIdx);

    thrust::fill(pressureForcePtr, pressureForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(viscousForcePtr, viscousForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(surfTenForcePtr, surfTenForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(externalForcePtr, externalForcePtr+particle.numParticles, make_float3(0.0f, 0.0f,0.0f));
    thrust::fill(totalForcePtr, totalForcePtr+particle.numParticles, make_float3(0.0f,0.0f,0.0f));
//    thrust::fill(densityErrPtr, densityErrPtr+particle.numParticles, 0.0f);
    thrust::fill(densityPtr, densityPtr+particle.numParticles, 0.0f);
    thrust::fill(pressurePtr, pressurePtr+particle.numParticles, 0.0f);
    thrust::fill(hashPtr, hashPtr+particle.numParticles, 0);
    thrust::fill(cellOccPtr, cellOccPtr+numCells, 0);
    thrust::fill(cellPartIdxPtr, cellPartIdxPtr+numCells, 0);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ResetTotalForce(float3 *totalForce, const uint numPoints)
{
    thrust::device_ptr<float3> totalForcePtr = thrust::device_pointer_cast(totalForce);
    thrust::fill(totalForcePtr, totalForcePtr+numPoints, make_float3(0.0f,0.0f,0.0f));
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ParticleHash(ParticleGpuData particle, const uint gridRes, const float cellWidth)
{
    uint numBlocks = iDivUp(particle.numParticles, 1024u);
    sphGPU_Kernels::ParticleHash_Kernel<<<numBlocks, 1024u>>>(particle, gridRes, cellWidth);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::SortParticlesByHash(ParticleGpuData particle)
{
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(particle.hash);
    thrust::device_ptr<float3> posPtr = thrust::device_pointer_cast(particle.pos);
    thrust::device_ptr<float3> velPtr = thrust::device_pointer_cast(particle.vel);
    thrust::device_ptr<uint> particleIdPtr = thrust::device_pointer_cast(particle.id);

    thrust::sort_by_key(hashPtr,
                        hashPtr + particle.numParticles,
                        thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr, particleIdPtr)));
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::SortParticlesByHash(AlgaeGpuData particle)
{
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(particle.hash);
    thrust::device_ptr<float3> posPtr = thrust::device_pointer_cast(particle.pos);
    thrust::device_ptr<float3> velPtr = thrust::device_pointer_cast(particle.vel);
    thrust::device_ptr<uint> particleIdPtr = thrust::device_pointer_cast(particle.id);
    thrust::device_ptr<float> prevPressPtr = thrust::device_pointer_cast(particle.prevPressure);
    thrust::device_ptr<float> illumPtr = thrust::device_pointer_cast(particle.illum);

    thrust::sort_by_key(hashPtr,
                        hashPtr + particle.numParticles,
                        thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr, particleIdPtr, prevPressPtr, illumPtr)));
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeParticleScatterIds(uint *cellOccupancy, uint *cellParticleIdx, const uint numCells)
{
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(cellOccupancy);
    thrust::device_ptr<uint> cellPartIdxPtr = thrust::device_pointer_cast(cellParticleIdx);

    thrust::exclusive_scan(cellOccPtr, cellOccPtr+numCells, cellPartIdxPtr);
}

void sphGPU::ComputeMaxCellOccupancy(uint *cellOccupancy, const uint numCells, unsigned int &_maxCellOcc)
{
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(cellOccupancy);

    _maxCellOcc = *thrust::max_element(cellOccPtr, cellOccPtr+numCells);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeParticleVolume(RigidGpuData particle, const uint gridRes)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeVolume_kernel<<<gridDim, blockSize>>>(particle);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeDensity(ParticleGpuData particle, const uint gridRes, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeDensity_kernel<<<gridDim, blockSize>>>(particle, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeDensityFluidFluid(ParticleGpuData particle, ParticleGpuData contributerParticle, const uint gridRes, const bool accumulate)
{

    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeDensityFluidFluid_kernel<<<gridDim, blockSize>>>(particle, contributerParticle, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------


void sphGPU::ComputeDensityFluidRigid(ParticleGpuData particle, RigidGpuData rigidParticle, const uint gridRes, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeDensityFluidRigid_kernel<<<gridDim, blockSize>>>(particle, rigidParticle, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputePressureFluid(FluidGpuData particle, const uint gridRes)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressure_kernel<<<gridDim, blockSize>>>(particle);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::SamplePressure(ParticleGpuData particleData, ParticleGpuData particleContributerData, const uint gridRes)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particleData.maxCellOcc, 1024u);

    sphGPU_Kernels::SamplePressure<<<gridDim, blockSize>>>(particleData, particleContributerData);
}


//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputePressureForceFluid(ParticleGpuData particleData, const uint gridRes, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particleData.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressureForce_kernel<<<gridDim, blockSize>>>(particleData, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputePressureForceFluidFluid(ParticleGpuData particle, ParticleGpuData contributerParticle, const uint gridRes, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressureForceFluidFluid_kernel<<<gridDim, blockSize>>>(particle, contributerParticle, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputePressureForceFluidRigid(ParticleGpuData particle, RigidGpuData rigidParticle, const uint gridRes, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particle.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressureForceFluidRigid_kernel<<<gridDim, blockSize>>>(particle, rigidParticle, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeViscForce(FluidGpuData particleData, const uint gridRes)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particleData.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeViscousForce_kernel<<<gridDim, blockSize>>>(particleData);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeSurfaceTensionForce(FluidGpuData particleData, const uint gridRes)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::max(std::min(particleData.maxCellOcc, 1024u), 32u);

    sphGPU_Kernels::ComputeSurfaceTensionForce_kernel<<<gridDim, blockSize>>>(particleData);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeForce(FluidGpuData particleData, const uint gridRes, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::max(std::min(particleData.maxCellOcc, 1024u), 27u);

    sphGPU_Kernels::ComputeForce_kernel<<<gridDim, blockSize>>>(particleData, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::ComputeTotalForce(FluidGpuData particleData,
                               const uint gridRes,
                               const bool accumulatePressure,
                               const bool accumulateViscous,
                               const bool accumulateSurfTen,
                               const bool accumulateExternal,
                               const bool accumulateGravity)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particleData.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeTotalForce_kernel<<<gridDim, blockSize>>>(accumulatePressure,
                                                                     accumulateViscous,
                                                                     accumulateSurfTen,
                                                                     accumulateExternal,
                                                                     accumulateGravity,
                                                                     particleData);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::Integrate(ParticleGpuData particleData, const uint gridRes, const float _dt)
{
    uint numBlocks = iDivUp(particleData.numParticles, 1024u);

    sphGPU_Kernels::Integrate_kernel<<<numBlocks, 1024u>>>(particleData, _dt);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::HandleBoundaries(ParticleGpuData particleData, const uint gridRes, const float _gridDim)
{
    uint numBlocks = iDivUp(particleData.numParticles, 1024u);

    sphGPU_Kernels::HandleBoundaries_Kernel<<<numBlocks, 1024u>>>(particleData, _gridDim);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::InitFluidAsCube(ParticleGpuData particle, const uint numPartsPerAxis, const float scale)
{

    uint threadsPerBlock = 8;
    dim3 blockDim(threadsPerBlock,threadsPerBlock,threadsPerBlock);
    uint blocksPerGrid = iDivUp(numPartsPerAxis,threadsPerBlock);
    dim3 gridDim(blocksPerGrid, blocksPerGrid, blocksPerGrid);

    sphGPU_Kernels::InitParticleAsCube_Kernel<<<gridDim, blockDim>>>(particle, numPartsPerAxis, scale);

}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::InitAlgaeIllumination(float *illum,
                                   const unsigned int numPoints)
{
    thrust::device_ptr<float> illumPtr = thrust::device_pointer_cast(illum);

    thrust::fill(illumPtr, illumPtr+numPoints, 0.0f);
}

//--------------------------------------------------------------------------------------------------------------------

void sphGPU::InitSphParticleIds(unsigned int *particleId,
                                const unsigned int numPoints)
{
    thrust::device_ptr<unsigned int> particleIdPtr = thrust::device_pointer_cast(particleId);
    thrust::sequence(particleIdPtr, particleIdPtr+numPoints, 0);
}

//--------------------------------------------------------------------------------------------------------------------
// Algae functions
void sphGPU::ComputeAdvectionForce(ParticleGpuData particleData, FluidGpuData advectorParticleData, const uint gridRes, const bool accumulate)
{

    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particleData.maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeAdvectionForce<<<gridDim, blockSize>>>(particleData, advectorParticleData, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------
void sphGPU::AdvectParticle(ParticleGpuData particleData, FluidGpuData advectorParticleData, const uint gridRes, const float deltaTime)
{

    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(particleData.maxCellOcc, 1024u);

    sphGPU_Kernels::AdvectParticle<<<gridDim, blockSize>>>(particleData, advectorParticleData, deltaTime);
}

//--------------------------------------------------------------------------------------------------------------------
void sphGPU::ComputeBioluminescence(AlgaeGpuData particleData, const uint gridRes, const float deltaTime)
{
    uint numBlocks = iDivUp(particleData.numParticles, 1024u);

    sphGPU_Kernels::ComputeBioluminescence<<<numBlocks, 1024u>>>(particleData, deltaTime);
}


//--------------------------------------------------------------------------------------------------------------------
