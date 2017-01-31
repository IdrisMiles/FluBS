
#include "../include/sphsolverGPU.h"
#include "sphGPU_Kernels.cuh"
//#include "../cuda_inc/vec_ops.cuh"
//#include "../cuda_inc/smoothingKernel.cuh"

#include <thrust/extrema.h>

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <algorithm>








uint iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


SPHSolverGPU::SPHSolverGPU(FluidProperty *_fluidProperty, FluidSolverProperty *_fluidSolverProperty)
{
    m_fluidProperty = _fluidProperty;
    m_fluidSolverProperty = _fluidSolverProperty;

    d_mass.resize(m_fluidProperty->numParticles);
    d_densities.resize(m_fluidProperty->numParticles);
    d_pressures.resize(m_fluidProperty->numParticles);
    d_pressureForces.resize(m_fluidProperty->numParticles);
    d_viscousForces.resize(m_fluidProperty->numParticles);
    d_surfaceTensionForces.resize(m_fluidProperty->numParticles);
    d_externalForces.resize(m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution);
    d_externalAcceleration.resize(m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution);
    d_gravityAcceleration.resize(m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution);
    d_totalForces.resize(m_fluidProperty->numParticles);
    d_particleHashIds.resize(m_fluidProperty->numParticles);
    d_cellOccupancy.resize(m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution);
    d_cellParticleIdx.resize(m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridResolution);

    ResetDevicePointers();

    m_threadsPerBlock = 1024;
    m_numBlock = iDivUp(m_fluidProperty->numParticles, m_threadsPerBlock);
}


SPHSolverGPU::~SPHSolverGPU()
{

}

FluidSolverProperty *SPHSolverGPU::GetFluidSolverProperty()
{
    return m_fluidSolverProperty;
}

void SPHSolverGPU::Solve(float _dt, float3* _d_p, float3* _d_v, float *_d_d)
{
    d_positions_ptr = _d_p;
    d_velocities_ptr = _d_v;
    d_densities_ptr = _d_d;

    ResetProperties();


    cudaThreadSynchronize();

    // Get particle hash IDs
    ParticleHash(d_particleHashIds_ptr, d_cellOccupancy_ptr, d_positions_ptr, m_fluidProperty->numParticles, m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridCellWidth, m_fluidProperty->numParticles);
    cudaThreadSynchronize();

    // Sort particles
    thrust::sort_by_key(d_particleHashIds.begin(), d_particleHashIds.end(), thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_positions_ptr), thrust::device_pointer_cast(d_velocities_ptr))));

    // Get Cell particle indexes - scatter addresses
    thrust::exclusive_scan(d_cellOccupancy.begin(), d_cellOccupancy.end(), d_cellParticleIdx.begin());

    uint maxCellOcc = *thrust::max_element(d_cellOccupancy.begin(), d_cellOccupancy.end());
    cudaThreadSynchronize();

    if(maxCellOcc > 1024u)
    {
        std::cout<<"Too many neighs\n";
    }

    // Run SPH solver
    ComputePressure(maxCellOcc, d_pressures_ptr, d_densities_ptr, m_fluidProperty->restDensity, m_fluidProperty->gasStiffness, d_mass_ptr, d_cellOccupancy_ptr, d_cellParticleIdx_ptr, d_positions_ptr, m_fluidProperty->numParticles, m_fluidSolverProperty->smoothingLength);
    cudaThreadSynchronize();

    // pressure force
    ComputePressureForce(maxCellOcc, d_pressureForces_ptr, d_pressures_ptr, d_densities_ptr, d_mass_ptr, d_positions_ptr, d_cellOccupancy_ptr, d_cellParticleIdx_ptr, m_fluidProperty->numParticles, m_fluidSolverProperty->smoothingLength);

    // TODO:
    // Compute boundary pressures


    // viscous force
    ComputeViscForce(maxCellOcc, d_viscousForces_ptr, m_fluidProperty->viscosity, d_velocities_ptr, d_densities_ptr, d_mass_ptr, d_positions_ptr, d_cellOccupancy_ptr, d_cellParticleIdx_ptr, m_fluidProperty->numParticles, m_fluidSolverProperty->smoothingLength);

    // Compute surface tension
    ComputeSurfaceTensionForce(maxCellOcc, d_surfaceTensionForces_ptr, m_fluidProperty->surfaceTension, m_fluidProperty->surfaceThreshold, d_densities_ptr, d_mass_ptr, d_positions_ptr, d_cellOccupancy_ptr, d_cellParticleIdx_ptr, m_fluidProperty->numParticles, m_fluidSolverProperty->smoothingLength);
    cudaThreadSynchronize();

    // Compute total force and acceleration
    ComputeTotalForce(maxCellOcc, d_totalForces_ptr, d_externalForces_ptr, d_pressureForces_ptr, d_viscousForces_ptr, d_surfaceTensionForces_ptr, d_mass_ptr, d_positions_ptr, d_velocities_ptr,d_cellOccupancy_ptr, d_cellParticleIdx_ptr, m_fluidProperty->numParticles, m_fluidSolverProperty->smoothingLength);
    cudaThreadSynchronize();

    // integrate particle position and velocities
    Integrate(maxCellOcc, d_totalForces_ptr, d_positions_ptr, d_velocities_ptr, _dt, m_fluidProperty->numParticles);
    cudaThreadSynchronize();

    // Handle boundaries
    HandleBoundaries(maxCellOcc, d_positions_ptr, d_velocities_ptr, (float)0.5f*m_fluidSolverProperty->gridCellWidth * m_fluidSolverProperty->gridResolution, m_fluidProperty->numParticles);
    cudaThreadSynchronize();



    //PrintInfo();

}

void SPHSolverGPU::ResetProperties()
{
    thrust::fill(d_pressureForces.begin(), d_pressureForces.end(), make_float3(0.0f,0.0f,0.0f));
    thrust::fill(d_viscousForces.begin(), d_viscousForces.end(), make_float3(0.0f,0.0f,0.0f));
    thrust::fill(d_surfaceTensionForces.begin(), d_surfaceTensionForces.end(), make_float3(0.0f,0.0f,0.0f));
    thrust::fill(d_externalForces.begin(), d_externalForces.end(), make_float3(0.0f, 0.0f,0.0f));
    thrust::fill(d_externalAcceleration.begin(), d_externalAcceleration.end(), make_float3(0.0f, 0.0f, 0.0f));
    thrust::fill(d_gravityAcceleration.begin(), d_gravityAcceleration.end(), make_float3(0.0f, -9.8f, 0.0f));
    thrust::fill(d_totalForces.begin(), d_totalForces.end(), make_float3(0.0f,0.0f,0.0f));
    thrust::fill(thrust::device_pointer_cast(d_densities_ptr), thrust::device_pointer_cast(d_densities_ptr)+m_fluidProperty->numParticles, 0.0f);
    thrust::fill(d_pressures.begin(), d_pressures.end(), 0.0f);
    thrust::fill(d_mass.begin(), d_mass.end(), m_fluidProperty->particleMass);
    thrust::fill(d_particleHashIds.begin(), d_particleHashIds.end(), 0u);
    thrust::fill(d_cellOccupancy.begin(), d_cellOccupancy.end(), 0u);
    thrust::fill(d_cellParticleIdx.begin(), d_cellParticleIdx.end(), 0u);
}

void SPHSolverGPU::ResetDevicePointers()
{
    d_mass_ptr = thrust::raw_pointer_cast(&d_mass[0]);
    d_densities_ptr = thrust::raw_pointer_cast(&d_densities[0]);
    d_pressures_ptr = thrust::raw_pointer_cast(&d_pressures[0]);
    d_pressureForces_ptr = thrust::raw_pointer_cast(&d_pressureForces[0]);
    d_viscousForces_ptr = thrust::raw_pointer_cast(&d_viscousForces[0]);
    d_surfaceTensionForces_ptr = thrust::raw_pointer_cast(&d_surfaceTensionForces[0]);
    d_externalForces_ptr = thrust::raw_pointer_cast(&d_externalForces[0]);
    d_externalAcceleration_ptr = thrust::raw_pointer_cast(&d_externalAcceleration[0]);
    d_gravityAcceleration_ptr = thrust::raw_pointer_cast(&d_gravityAcceleration[0]);
    d_totalForces_ptr = thrust::raw_pointer_cast(&d_totalForces[0]);
    d_particleHashIds_ptr = thrust::raw_pointer_cast(&d_particleHashIds[0]);
    d_cellOccupancy_ptr = thrust::raw_pointer_cast(d_cellOccupancy.data());
    d_cellParticleIdx_ptr = thrust::raw_pointer_cast(&d_cellParticleIdx[0]);
}

void SPHSolverGPU::ParticleHash(uint *hash, uint *cellOcc, float3 *particles, const uint N, const uint gridRes, const float cellWidth, const uint numPoints)
{
    uint numBlocks = iDivUp(numPoints, m_threadsPerBlock);
    sphGPU_Kernels::ParticleHash_Kernel<<<numBlocks, m_threadsPerBlock>>>(hash, cellOcc, particles, N, gridRes, cellWidth);
}

void SPHSolverGPU::ComputePressure(const uint maxCellOcc, float *pressure, float *density, const float restDensity, const float gasConstant, const float *mass, const uint *cellOcc, const uint *cellPartIdx, const float3 *particles, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressure_kernel<<<gridDim, blockSize>>>(pressure, density, restDensity, gasConstant, mass, cellOcc, cellPartIdx, particles, numPoints, smoothingLength);
}

void SPHSolverGPU::ComputePressureForce(const uint maxCellOcc, float3 *pressureForce, const float *pressure, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputePressureForce_kernel<<<gridDim, blockSize>>>(pressureForce, pressure, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void SPHSolverGPU::ComputeViscForce(const uint maxCellOcc, float3 *viscForce, const float viscCoeff, const float3 *velocity, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeViscousForce_kernel<<<gridDim, blockSize>>>(viscForce, viscCoeff, velocity, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void SPHSolverGPU::ComputeSurfaceTensionForce(const uint maxCellOcc, float3 *surfTenForce, const float surfaceTension, const float surfaceThreshold, /*const */float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeSurfaceTensionForce_kernel<<<gridDim, blockSize>>>(surfTenForce, surfaceTension, surfaceThreshold, density, mass, particles, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void SPHSolverGPU::ComputeTotalForce(const uint maxCellOcc, float3 *force, const float3 *externalForce, const float3 *pressureForce, const float3 *viscForce, const float3 *surfaceTensionForce, const float *mass, const float3 *particles, const float3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{
    dim3 gridDim = dim3(m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution, m_fluidSolverProperty->gridResolution);
    uint blockSize = std::min(maxCellOcc, 1024u);

    sphGPU_Kernels::ComputeForces_kernel<<<gridDim, blockSize>>>(force, externalForce, pressureForce, viscForce, surfaceTensionForce, mass, particles, velocities, cellOcc, cellPartIdx, numPoints, smoothingLength);
}

void SPHSolverGPU::Integrate(const uint maxCellOcc, float3 *force, float3 *particles, float3 *velocities, const float _dt, const uint numPoints)
{
    uint numBlocks = iDivUp(numPoints, m_threadsPerBlock);

    sphGPU_Kernels::Integrate_kernel<<<numBlocks, m_threadsPerBlock>>>(force, particles, velocities, _dt, numPoints);
}

void SPHSolverGPU::HandleBoundaries(const uint maxCellOcc, float3 *particles, float3 *velocities, const float _gridDim, const uint numPoints)
{
    uint numBlocks = iDivUp(numPoints, m_threadsPerBlock);

    sphGPU_Kernels::HandleBoundaries_Kernel<<<numBlocks, m_threadsPerBlock>>>(particles, velocities, _gridDim, numPoints);
}

void SPHSolverGPU::InitFluidAsCube(float3 *particles, float3 *velocities, float *densities, const float restDensity, const unsigned int numParticles, const unsigned int numPartsPerAxis, const float scale)
{

    uint threadsPerBlock = 8;
    dim3 blockDim(threadsPerBlock,threadsPerBlock,threadsPerBlock);
    uint blocksPerGrid = iDivUp(numPartsPerAxis,threadsPerBlock);
    dim3 gridDim(blocksPerGrid, blocksPerGrid, blocksPerGrid);

    sphGPU_Kernels::InitParticleAsCube_Kernel<<<gridDim, blockDim>>>(particles, velocities, densities, restDensity, numParticles, numPartsPerAxis, scale);

}


void SPHSolverGPU::PrintInfo()
{
    std::cout << "\n\nHash: \n";
    thrust::copy(d_particleHashIds.begin(), d_particleHashIds.end(), std::ostream_iterator<uint>(std::cout, " "));

    std::cout << "\n\nOccupancy: \n";
    thrust::copy(d_cellOccupancy.begin(), d_cellOccupancy.end(), std::ostream_iterator<uint>(std::cout, " "));

    std::cout << "\n\nCell particle Ids: \n";
    thrust::copy(d_cellParticleIdx.begin(), d_cellParticleIdx.end(), std::ostream_iterator<uint>(std::cout, " "));

    std::cout << "\n\nPressure: \n";
    thrust::copy(d_pressures.begin(), d_pressures.end(), std::ostream_iterator<float>(std::cout, " "));
}
