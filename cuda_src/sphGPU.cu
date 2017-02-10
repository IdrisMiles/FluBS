#include "../include/SPH/sphGPU.h"
#include "../cuda_inc/sphGPU_Kernels.cuh"


#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include <algorithm>

uint sphGPU::iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}

void sphGPU::ResetProperties(float3 *pressureForce,
                             float3 *viscousForce,
                             float3 *surfTenForce,
                             float3 *externalForce,
                             float3 *totalForce,
                             float * mass,
                             float *density,
                             float *pressure,
                             uint *hash,
                             uint *cellOcc,
                             uint *cellPartIdx,
                             const float massValue,
                             const uint numCells,
                             const uint numPoints)
{
    thrust::device_ptr<float3> pressureForcePtr = thrust::device_pointer_cast(pressureForce);
    thrust::device_ptr<float3> viscousForcePtr = thrust::device_pointer_cast(viscousForce);
    thrust::device_ptr<float3> surfTenForcePtr = thrust::device_pointer_cast(surfTenForce);
    thrust::device_ptr<float3> externalForcePtr = thrust::device_pointer_cast(externalForce);
    thrust::device_ptr<float3> totalForcePtr = thrust::device_pointer_cast(totalForce);
    thrust::device_ptr<float> massPtr = thrust::device_pointer_cast(mass);
    thrust::device_ptr<float> densityPtr = thrust::device_pointer_cast(density);
    thrust::device_ptr<float> pressurePtr = thrust::device_pointer_cast(pressure);
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(hash);
    thrust::device_ptr<uint> cellOccPtr = thrust::device_pointer_cast(cellOcc);
    thrust::device_ptr<uint> cellPartIdxPtr = thrust::device_pointer_cast(cellPartIdx);

    thrust::fill(pressureForcePtr, pressureForcePtr+numPoints, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(viscousForcePtr, viscousForcePtr+numPoints, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(surfTenForcePtr, surfTenForcePtr+numPoints, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(externalForcePtr, externalForcePtr+numPoints, make_float3(0.0f, 0.0f,0.0f));
    thrust::fill(totalForcePtr, totalForcePtr+numPoints, make_float3(0.0f,0.0f,0.0f));
    thrust::fill(massPtr, massPtr+numPoints, massValue);
    thrust::fill(densityPtr, densityPtr+numPoints, 0.0f);
    thrust::fill(pressurePtr, pressurePtr+numPoints, 0.0f);
    thrust::fill(hashPtr, hashPtr+numPoints, 0u);
    thrust::fill(cellOccPtr, cellOccPtr+numCells, 0u);
    thrust::fill(cellPartIdxPtr, cellPartIdxPtr+numCells, 0u);
}

void sphGPU::ParticleHash(uint *hash, uint *cellOcc, float3 *particles, const uint numPoints, const uint gridRes, const float cellWidth)
{
    uint numBlocks = iDivUp(numPoints, 1024u);
    sphGPU_Kernels::ParticleHash_Kernel<<<numBlocks, 1024u>>>(hash, cellOcc, particles, numPoints, gridRes, cellWidth);
}

void sphGPU::SortParticlesByHash(uint *hash, float3 *position, float3 *velocity, const uint numPoints)
{
    thrust::device_ptr<uint> hashPtr = thrust::device_pointer_cast(hash);
    thrust::device_ptr<float3> posPtr = thrust::device_pointer_cast(position);
    thrust::device_ptr<float3> velPtr = thrust::device_pointer_cast(velocity);

    thrust::sort_by_key(hashPtr,
                        hashPtr + numPoints,
                        thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr)));
}

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

void sphGPU::ComputeParticleVolume(const uint maxCellOcc,
                                   const uint gridRes,
                                   float *volume,
                                   const uint *cellOcc,
                                   const uint *cellPartIdx,
                                   const float3 *particles,
                                   const uint numPoints,
                                   const float smoothingLength)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeVolume_kernel<<<gridDim, blockSize>>>(volume, cellOcc, cellPartIdx, particles, numPoints, smoothingLength);
}

void sphGPU::ComputeDensity(const uint maxCellOcc,
                            const uint gridRes,
                            float *density,
                            const float *mass,
                            const uint *cellOcc,
                            const uint *cellPartIdx,
                            const float3 *particles,
                            const uint numPoints,
                            const float smoothingLength,
                            const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeDensity_kernel<<<gridDim, blockSize>>>(density, mass, cellOcc, cellPartIdx, particles, numPoints, smoothingLength, accumulate);
}

void sphGPU::ComputeDensityFluidRigid(const uint maxCellOcc,
                              const uint gridRes,
                              const uint numPoints,
                              const float fluidRestDensity,
                              float *fluidDensity,
                              const float3 *fluidPos,
                              const uint *fluidCellOcc,
                              const uint *fluidCellPartIdx,
                              float *rigidVolume,
                              const float3 *rigidPos,
                              const uint *rigidCellOcc,
                              const uint *rigidCellPartIdx,
                              const float smoothingLength,
                              const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeDensityFluidRigid_kernel<<<gridDim, blockSize>>>(numPoints, fluidRestDensity, fluidDensity, fluidCellOcc, fluidCellPartIdx, fluidPos, rigidVolume, rigidCellOcc, rigidCellPartIdx, rigidPos, smoothingLength, accumulate);
}

void sphGPU::ComputePressure(const uint maxCellOcc,
                             const uint gridRes,
                             float *pressure,
                             float *density,
                             const float restDensity,
                             const float gasConstant,
                             const uint *cellOcc,
                             const uint *cellPartIdx,
                             const uint numPoints)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressure_kernel<<<gridDim, blockSize>>>(pressure, density, restDensity, gasConstant, cellOcc, cellPartIdx, numPoints);
}

void sphGPU::ComputePressureForce(const uint maxCellOcc, const uint gridRes, float3 *pressureForce, const float *pressure, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength, const bool accumulate)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressureForce_kernel<<<gridDim, blockSize>>>(pressureForce, pressure, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength, accumulate);
}

void sphGPU::ComputeViscForce(const uint maxCellOcc, const uint gridRes, float3 *viscForce, const float viscCoeff, const float3 *velocity, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeViscousForce_kernel<<<gridDim, blockSize>>>(viscForce, viscCoeff, velocity, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void sphGPU::ComputeSurfaceTensionForce(const uint maxCellOcc, const uint gridRes, float3 *surfTenForce, const float surfaceTension, const float surfaceThreshold, /*const */float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeSurfaceTensionForce_kernel<<<gridDim, blockSize>>>(surfTenForce, surfaceTension, surfaceThreshold, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void sphGPU::ComputeTotalForce(const uint maxCellOcc, const uint gridRes, float3 *force, const float3 *externalForce, const float3 *pressureForce, const float3 *viscForce, const float3 *surfaceTensionForce, const float *mass, const float3 *particles, const float3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeForces_kernel<<<gridDim, blockSize>>>(force, externalForce, pressureForce, viscForce, surfaceTensionForce, mass, particles, velocities, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void sphGPU::Integrate(const uint maxCellOcc, const uint gridRes, float3 *force, float3 *particles, float3 *velocities, const float _dt, const uint numPoints)
{
    uint numBlocks = iDivUp(numPoints, 1024u);

    sphGPU_Kernels::Integrate_kernel<<<numBlocks, 1024u>>>(force, particles, velocities, _dt, numPoints);
}

void sphGPU::HandleBoundaries(const uint maxCellOcc, const uint gridRes, float3 *particles, float3 *velocities, const float _gridDim, const uint numPoints)
{
    uint numBlocks = iDivUp(numPoints, 1024u);

    sphGPU_Kernels::HandleBoundaries_Kernel<<<numBlocks, 1024u>>>(particles, velocities, _gridDim, numPoints);
}

void sphGPU::InitFluidAsCube(float3 *particles, float3 *velocities, float *densities, const float restDensity, const uint numParticles, const uint numPartsPerAxis, const float scale)
{

    uint threadsPerBlock = 8;
    dim3 blockDim(threadsPerBlock,threadsPerBlock,threadsPerBlock);
    uint blocksPerGrid = iDivUp(numPartsPerAxis,threadsPerBlock);
    dim3 gridDim(blocksPerGrid, blocksPerGrid, blocksPerGrid);

    sphGPU_Kernels::InitParticleAsCube_Kernel<<<gridDim, blockDim>>>(particles, velocities, densities, restDensity, numParticles, numPartsPerAxis, scale);

}