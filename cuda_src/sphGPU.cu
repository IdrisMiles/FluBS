#include "../include/sphGPU.h"
#include "../cuda_inc/sphGPU_Kernels.cuh"

//uint sphGPU::iDivUp(uint a, uint b)
//{
//    uint c = a/b;
//    c += (a%b == 0) ? 0: 1;
//    return c;
//}



//void sphGPU::ParticleHash(uint *hash, uint *cellOcc, float3 *particles, const uint numPoints, const uint gridRes, const float cellWidth)
//{
//    uint numBlocks = iDivUp(numPoints, 1024u);
//    sphGPU_Kernels::ParticleHash_Kernel<<<numBlocks, 1024u>>>(hash, cellOcc, particles, numPoints, gridRes, cellWidth);
//}

//void sphGPU::ComputePressure(const uint maxCellOcc, const uint gridRes, float *pressure, float *density, const float restDensity, const float gasConstant, const float *mass, const uint *cellOcc, const uint *cellPartIdx, const float3 *particles, const uint numPoints, const float smoothingLength)
//{
//    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
//    uint blockSize = std::min(maxCellOcc, 1024u);

//    sphGPU_Kernels::ComputePressure_kernel<<<gridDim, blockSize>>>(pressure, density, restDensity, gasConstant, mass, cellOcc, cellPartIdx, particles, numPoints, smoothingLength);
//}

//void sphGPU::ComputePressureForce(const uint maxCellOcc, const uint gridRes, float3 *pressureForce, const float *pressure, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
//{
//    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
//    uint blockSize = std::min(maxCellOcc, 1024u);

//    sphGPU_Kernels::ComputePressureForce_kernel<<<gridDim, blockSize>>>(pressureForce, pressure, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
//}

//void sphGPU::ComputeViscForce(const uint maxCellOcc, const uint gridRes, float3 *viscForce, const float viscCoeff, const float3 *velocity, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
//{
//    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
//    uint blockSize = std::min(maxCellOcc, 1024u);

//    sphGPU_Kernels::ComputeViscousForce_kernel<<<gridDim, blockSize>>>(viscForce, viscCoeff, velocity, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
//}

//void sphGPU::ComputeSurfaceTensionForce(const uint maxCellOcc, const uint gridRes, float3 *surfTenForce, const float surfaceTension, const float surfaceThreshold, /*const */float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
//{
//    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
//    uint blockSize = std::min(maxCellOcc, 1024u);

//    sphGPU_Kernels::ComputeSurfaceTensionForce_kernel<<<gridDim, blockSize>>>(surfTenForce, surfaceTension, surfaceThreshold, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
//}

//void sphGPU::ComputeTotalForce(const uint maxCellOcc, const uint gridRes, float3 *force, const float3 *externalForce, const float3 *pressureForce, const float3 *viscForce, const float3 *surfaceTensionForce, const float *mass, const float3 *particles, const float3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
//{
//    dim3 gridDim = dim3(gridRes, gridRes, gridRes);
//    uint blockSize = std::min(maxCellOcc, 1024u);

//    sphGPU_Kernels::ComputeForces_kernel<<<gridDim, blockSize>>>(force, externalForce, pressureForce, viscForce, surfaceTensionForce, mass, particles, velocities, cellOcc, cellPartIdx, numPoints, smoothingLength);
//}

//void sphGPU::Integrate(const uint maxCellOcc, const uint gridRes, float3 *force, float3 *particles, float3 *velocities, const float _dt, const uint numPoints)
//{
//    uint numBlocks = iDivUp(numPoints, 1024u);

//    sphGPU_Kernels::Integrate_kernel<<<numBlocks, 1024u>>>(force, particles, velocities, _dt, numPoints);
//}

//void sphGPU::HandleBoundaries(const uint maxCellOcc, const uint gridRes, float3 *particles, float3 *velocities, const float _gridDim, const uint numPoints)
//{
//    uint numBlocks = iDivUp(numPoints, 1024u);

//    sphGPU_Kernels::HandleBoundaries_Kernel<<<numBlocks, 1024u>>>(particles, velocities, _gridDim, numPoints);
//}

//void sphGPU::InitFluidAsCube(float3 *particles, float3 *velocities, float *densities, const float restDensity, const uint numParticles, const uint numPartsPerAxis, const float scale)
//{

//    uint threadsPerBlock = 8;
//    dim3 blockDim(threadsPerBlock,threadsPerBlock,threadsPerBlock);
//    uint blocksPerGrid = iDivUp(numPartsPerAxis,threadsPerBlock);
//    dim3 gridDim(blocksPerGrid, blocksPerGrid, blocksPerGrid);

//    sphGPU_Kernels::InitParticleAsCube_Kernel<<<gridDim, blockDim>>>(particles, velocities, densities, restDensity, numParticles, numPartsPerAxis, scale);

//}
