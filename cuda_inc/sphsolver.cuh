#ifndef SPHSOLVER_CUH
#define SPHSOLVER_CUH




__device__ float SmoothingKernel_Kernel(const float &_r, const float &_h, const float &_smoothingConstant = 1.0f);
__global__ void ParticleHash_Kernel(unsigned int *hash, unsigned int *cellOcc, const float3 *particles, const unsigned int N, const unsigned int gridRes, const float &cellWidth);
__global__ void SPHSolve_Kernel(unsigned int *hash, unsigned int *cellOcc, unsigned int *cellIds, const float3 *particles, const unsigned int numPoints, const unsigned int gridRes, const float smoothingLength);
__global__ void InitParticleAsCube_Kernel(float3 *particles, const unsigned int numParticles);
#endif // SPHSOLVER_CUH
