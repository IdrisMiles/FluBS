#ifndef SPHSOLVER_H
#define SPHSOLVER_H

// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <math_constants.h>

// Thrust includes for CUDA stuff
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

#include "fluidproperty.h"


class SPHSolverGPU
{

public:
    SPHSolverGPU(FluidProperty* _fluidProperty);
    virtual ~SPHSolverGPU();

    void Init();
    void Solve(float _dt, float3 *_d_p, float3 *_d_v, float *_d_d);

    void ParticleHash(unsigned int *hash, unsigned int *cellOcc, float3 *particles, const unsigned int N, const unsigned int gridRes, const float cellWidth);
    void ComputePressure(const uint maxCellOcc, float *pressure, float *density, const float restDensity, const float gasConstant, const float *mass, const uint *cellOcc, const uint *cellPartIdx, const float3 *particles, const uint numPoints, const float smoothingLength);
    void ComputePressureForce(const uint maxCellOcc, float3 *pressureForce, const float *pressure, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
    void ComputeViscForce(const uint maxCellOcc, float3 *viscForce, const float viscCoeff, const float3 *velocity, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
    void ComputeSurfaceTensionForce(const uint maxCellOcc, float3 *surfTenForce, const float surfaceTension, const float surfaceThreshold, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
    void ComputeTotalForce(const uint maxCellOcc, float3 *force, const float3 *externalForce, const float3 *pressureForce, const float3 *viscForce, const float3 *surfaceTensionForce, const float *mass, const float3 *particles, const float3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
    void Integrate(const uint maxCellOcc, float3 *force, float3 *particles, float3 *velocities, const float _dt, const uint numPoints);
    void HandleBoundaries(const uint maxCellOcc, float3 *particles, float3 *velocities, const float _gridDim, const uint numPoints);

    void SPHSolve(const unsigned int maxCellOcc, const unsigned int *cellOcc, const unsigned int *cellIds, float3 *particles, float3 *velocities, const unsigned int numPoints, const unsigned int gridRes, const float smoothingLength, const float dt);

    void InitParticleAsCube(float3 *particles, float3 *velocities, float *densities, const float restDensity, const unsigned int numParticles, const unsigned int numPartsPerAxis, const float scale);


private:
    FluidProperty* m_fluidProperty;

    thrust::device_vector<float3> d_positions;
    thrust::device_vector<float3> d_velocities;
    thrust::device_vector<float3> d_pressureForces;
    thrust::device_vector<float3> d_viscousForces;
    thrust::device_vector<float3> d_surfaceTensionForces;
    thrust::device_vector<float3> d_externalForces;
    thrust::device_vector<float3> d_totalForces;
    //thrust::device_vector<float> d_colourField;
    thrust::device_vector<float> d_densities;
    thrust::device_vector<float> d_pressures;
    thrust::device_vector<float> d_mass;
    thrust::device_vector<unsigned int> d_particleHashIds;
    thrust::device_vector<unsigned int> d_cellOccupancy;
    thrust::device_vector<unsigned int> d_cellParticleIdx;  // holds indexes into d_particles for start of each cell

    float3* d_positions_ptr;
    float3* d_velocities_ptr;
    float3* d_pressureForces_ptr;
    float3* d_viscousForces_ptr;
    float3* d_surfaceTensionForces_ptr;
    float3* d_externalForces_ptr;
    float3* d_totalForces_ptr;
    float* d_densities_ptr;
    float* d_pressures_ptr;
    float* d_mass_ptr;
    unsigned int* d_particleHashIds_ptr;
    unsigned int* d_cellOccupancy_ptr;
    unsigned int* d_cellParticleIdx_ptr;

    unsigned int m_threadsPerBlock;
    unsigned int m_numBlock;
    dim3 blockDim;
    dim3 gridDim;

};

#endif // SPHSOLVER_H
