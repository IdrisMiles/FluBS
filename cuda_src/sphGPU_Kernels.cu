#include "../cuda_inc/sphGPU_Kernels.cuh"


#include "../cuda_inc/vec_ops.cuh"
#include "../cuda_inc/smoothingKernel.cuh"

#include <functional>

#include <math_constants.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ParticleHash_Kernel(uint *hash,
                                                    uint *cellOcc,
                                                    const float3 *particles,
                                                    const uint N,
                                                    const uint gridRes,
                                                    const float cellWidth)
{
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Sanity check
    if (idx >= N)
    {
        return;
    }

    float gridDim = gridRes * cellWidth;
    float invGridDim = 1.0f / gridDim;
    float3 particle = particles[idx];
    uint hashID;

    // Get normalised particle positions [0-1]
    float normX = (particle.x + (0.5f * gridDim)) * invGridDim;
    float normY = (particle.y + (0.5f * gridDim)) * invGridDim;
    float normZ = (particle.z + (0.5f * gridDim)) * invGridDim;


    // Get hash values for x, y, z
    uint hashX = floor(normX * gridRes);
    uint hashY = floor(normY * gridRes);
    uint hashZ = floor(normZ * gridRes);

    hashX = (hashX >= gridRes) ? gridRes-1 : hashX;
    hashY = (hashY >= gridRes) ? gridRes-1 : hashY;
    hashZ = (hashZ >= gridRes) ? gridRes-1 : hashZ;

    hashX = (hashX < 0) ? 0 : hashX;
    hashY = (hashY < 0) ? 0 : hashY;
    hashZ = (hashZ < 0) ? 0 : hashZ;

    hashID = hashX + (hashY * gridRes) + (hashZ * gridRes * gridRes);

    if(hashID >= gridRes * gridRes * gridRes || hashID < 0)
    {
        printf("Hash out of bounds\n");
        printf("%u, %u, %u\n", hashX, hashY, hashZ);
    }

    // Update hash id for this particle
    hash[idx] = hashID;


    // Update cell occupancy for the cell
    atomicAdd(&cellOcc[hashID], 1u);


}




__global__ void sphGPU_Kernels::ParticleHash_Kernel(ParticleGpuData particle,
                                                    const uint gridRes,
                                                    const float cellWidth)
{
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Sanity check
    if (idx >= particle.numParticles)
    {
        return;
    }

    float gridDim = gridRes * cellWidth;
    float invGridDim = 1.0f / gridDim;
    float3 pos = particle.pos[idx];
    uint hashID;

    // Get normalised particle positions [0-1]
    float normX = (pos.x + (0.5f * gridDim)) * invGridDim;
    float normY = (pos.y + (0.5f * gridDim)) * invGridDim;
    float normZ = (pos.z + (0.5f * gridDim)) * invGridDim;


    // Get hash values for x, y, z
    uint hashX = floor(normX * gridRes);
    uint hashY = floor(normY * gridRes);
    uint hashZ = floor(normZ * gridRes);

    hashX = (hashX >= gridRes) ? gridRes-1 : hashX;
    hashY = (hashY >= gridRes) ? gridRes-1 : hashY;
    hashZ = (hashZ >= gridRes) ? gridRes-1 : hashZ;

    hashX = (hashX < 0) ? 0 : hashX;
    hashY = (hashY < 0) ? 0 : hashY;
    hashZ = (hashZ < 0) ? 0 : hashZ;

    hashID = hashX + (hashY * gridRes) + (hashZ * gridRes * gridRes);

    if(hashID >= gridRes * gridRes * gridRes || hashID < 0)
    {
        printf("Hash out of bounds\n");
        printf("%u, %u, %u\n", hashX, hashY, hashZ);
    }

    // Update hash id for this particle
    particle.hash[idx] = hashID;


    // Update cell occupancy for the cell
    atomicAdd(&particle.cellOcc[hashID], 1u);


}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeVolume_kernel(RigidGpuData particle)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;

    if(thisParticleGlobalIdx < particle.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);



        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = particle.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = particle.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();



        int neighLocalIdx;
        float accVolume = 0.0f;
        float3 thisParticle = particle.pos[thisParticleGlobalIdx];

        uint numNeighCells = 0;
        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    numNeighCells++;

                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        float3 neighParticle = particle.pos[neighParticleGlobalIdx];

                        accVolume += fabs(Poly6Kernel_Kernel(length(thisParticle - neighParticle), particle.smoothingLength));
                    }
                }
            }
        }

        accVolume = 1.0f / accVolume;

        if(isnan(accVolume) || fabs(accVolume) < FLT_EPSILON)
        {
            particle.volume[thisParticleGlobalIdx] = 1.0f;
        }
        else
        {
            particle.volume[thisParticleGlobalIdx] = 10.0f*accVolume;
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeDensity_kernel(ParticleGpuData particle, const bool accumulate)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;


    if((thisParticleGlobalIdx < particle.numParticles) && (threadIdx.x < thisCellOcc) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);


        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = particle.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = particle.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();


        int neighLocalIdx;
        float accDensity = 0.0f;
        float thisDensity = 0.0f;
        float thisMass = particle.mass;
        float3 thisParticle = particle.pos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    // Get density contribution from other fluid particles
                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighParticle = particle.pos[neighParticleGlobalIdx];

                        thisDensity = thisMass * Poly6Kernel_Kernel(length(thisParticle - neighParticle), particle.smoothingLength);

                        accDensity += thisDensity;
                    }
                }
            }
        }

        if(isnan(accDensity))
        {
            if(!accumulate)
            {
                particle.den[thisParticleGlobalIdx] = 0.0f;
            }
        }
        else
        {
            if(accumulate)
            {
                atomicAdd(&particle.den[thisParticleGlobalIdx], accDensity);
            }
            else
            {
                particle.den[thisParticleGlobalIdx] = accDensity;
            }
        }


    }

}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeDensityFluidRigid_kernel(ParticleGpuData particle, RigidGpuData rigidParticle, const bool accumulate)
{    
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;



    if((thisParticleGlobalIdx < particle.numParticles) && (threadIdx.x < thisCellOcc) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);


        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = rigidParticle.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = rigidParticle.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();


        int neighLocalIdx;
        float accDensity = 0.0f;
        float thisDensity = 0.0f;
        float3 thisParticle = particle.pos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    // Get density contribution from other fluid particles
                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighParticle = rigidParticle.pos[neighParticleGlobalIdx];

                        thisDensity = particle.restDen * rigidParticle.volume[neighParticleGlobalIdx] * Poly6Kernel_Kernel(length(thisParticle - neighParticle), particle.smoothingLength);

                        accDensity += (thisDensity);
                    }
                }
            }
        }

        if(isnan(accDensity))
        {
            if(!accumulate)
            {
                particle.den[thisParticleGlobalIdx] = 0.0f;
            }
        }
        else
        {
            if(accumulate)
            {
                atomicAdd(&particle.den[thisParticleGlobalIdx], accDensity);
            }
            else
            {
                particle.den[thisParticleGlobalIdx] = accDensity;
            }
        }

    } // end if valid point
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeDensityFluidFluid_kernel(ParticleGpuData particle, ParticleGpuData contributerParticle, const bool accumulate)
{    
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;


    if((thisParticleGlobalIdx < particle.numParticles) && (threadIdx.x < thisCellOcc) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);


        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = contributerParticle.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = contributerParticle.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();


        int neighLocalIdx;
        float accDensity = 0.0f;
        float thisDensity = 0.0f;
        float3 thisParticle = particle.pos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);

                    // Get density contribution from other fluid particles
                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighParticle = contributerParticle.pos[neighParticleGlobalIdx];

                        thisDensity = contributerParticle.mass * Poly6Kernel_Kernel(length(thisParticle - neighParticle), particle.smoothingLength);

                        accDensity += thisDensity;
                    }
                }
            }
        }

        if(isnan(accDensity))
        {
            if(!accumulate)
            {
                particle.den[thisParticleGlobalIdx] = 0.0f;
            }
        }
        else
        {
            if(accumulate)
            {
                atomicAdd(&particle.den[thisParticleGlobalIdx], accDensity);
            }
            else
            {
                particle.den[thisParticleGlobalIdx] = accDensity;
            }
        }

    } // end if valid point
}



//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputePressure_kernel(FluidGpuData particle)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;



    if(thisParticleGlobalIdx < particle.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        //float beta = 0.35;
        //float gamma = 7.0f;
        //float accPressure = beta * (pow((accDensity/restDensity), gamma)-1.0f);
        //float accPressure = gasConstant * ((accDensity/restDensity) - 1.0f);

//        float k = 50.0f;
        float gamma = 7.0f;
        float accPressure = (particle.gasStiffness*particle.restDen/ gamma) * (pow((particle.den[thisParticleGlobalIdx]/particle.restDen), gamma) - 1.0f);

//        float accPressure = gasConstant * (density[thisParticleGlobalIdx] - restDensity);
//        float accPressure = gasConstant * ( (density[thisParticleGlobalIdx] / restDensity) - 1.0f);

        if(isnan(accPressure))
        {
//            printf("nan pressure \n");
            particle.pressure[thisParticleGlobalIdx] = 0.0f;
        }
        else
        {
            particle.pressure[thisParticleGlobalIdx] = accPressure;
        }

    }

}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::SamplePressure(ParticleGpuData particleData, ParticleGpuData particleContributerData)
{

    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if((thisParticleGlobalIdx < particleData.numParticles) && (threadIdx.x < thisCellOcc) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);


        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = particleContributerData.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = particleContributerData.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();


        int neighLocalIdx;
        float accPressue = 0.0f;
        float3 thisPos = particleData.pos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighPos = particleContributerData.pos[neighParticleGlobalIdx];
                        float W = Poly6Kernel_Kernel(length(thisPos-neighPos), particleData.smoothingLength);
                        float invDen = 1.0f / particleContributerData.den[neighParticleGlobalIdx];
                        accPressue += invDen * W * particleContributerData.pressure[neighParticleGlobalIdx];
                    }
                }
            }
        }

        accPressue *= particleContributerData.mass;

        particleData.pressure[thisParticleGlobalIdx] = accPressue;
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputePressureForce_kernel(ParticleGpuData particleData, const bool accumulate)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {

        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);


        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = particleData.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = particleData.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();


        int neighLocalIdx;
        float3 accPressureForce = make_float3(0.0f, 0.0f, 0.0f);


        float thisMass = particleData.mass;
        float thisDensity = particleData.den[thisParticleGlobalIdx];
        float thisPressure = particleData.pressure[thisParticleGlobalIdx];
        float3 thisParticle = particleData.pos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx != thisParticleGlobalIdx)
                        {
                            float3 neighParticle = particleData.pos[neighParticleGlobalIdx];
                            float neighPressure = particleData.pressure[neighParticleGlobalIdx];
                            float neighDensity = particleData.den[neighParticleGlobalIdx];

//                            float pressOverDens = (fabs(neighDensity)<FLT_EPSILON ? 0.0f: (thisPressure + neighPressure) / (2.0f* neighDensity));

//                            accPressureForce = accPressureForce + (thisMass * pressOverDens * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, smoothingLength));
//                            accPressureForce = accPressureForce + (thishMass * (thisPressure+neighPressure) / (neighDensity + neighDensity) * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, smoothingLength));


                            accPressureForce = accPressureForce + ( ((thisPressure/(thisDensity*thisDensity)) + (neighPressure/(neighDensity*neighDensity))) * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, particleData.smoothingLength) );
                        }
                    }
                }
            }
        }


        if(!accumulate)
        {
            particleData.pressureForce[thisParticleGlobalIdx] = -1.0f * thisMass * thisMass * accPressureForce;
//            pressureForce[thisParticleGlobalIdx] = -1.0f * accPressureForce;
        }
        else
        {
            particleData.pressureForce[thisParticleGlobalIdx] = particleData.pressureForce[thisParticleGlobalIdx] + (-1.0f * thisMass * thisMass * accPressureForce);
//            pressureForce[thisParticleGlobalIdx] = pressureForce[thisParticleGlobalIdx] + (-1.0f * accPressureForce);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputePressureForceFluidFluid_kernel(ParticleGpuData particle, ParticleGpuData contributerParticle, const bool accumulate)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particle.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {

        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);



        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = contributerParticle.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = contributerParticle.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
        __syncthreads();


        int neighLocalIdx;
        float3 accPressureForce = make_float3(0.0f, 0.0f, 0.0f);


        float thisMass = particle.mass;
        float thisDensity = particle.den[thisParticleGlobalIdx];
        float thisPressure = particle.pressure[thisParticleGlobalIdx];
        float3 thisParticle = particle.pos[thisParticleGlobalIdx];
//        float neighMass = fluidContribMass;

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx != thisParticleGlobalIdx)
                        {
                            float3 neighParticle = contributerParticle.pos[neighParticleGlobalIdx];
                            float neighPressure = contributerParticle.pressure[neighParticleGlobalIdx];
                            float neighDensity = contributerParticle.den[neighParticleGlobalIdx];

//                            float pressOverDens = (fabs(neighDensity)<FLT_EPSILON ? 0.0f: (thisPressure + neighPressure) / (2.0f* neighDensity));

//                            accPressureForce = accPressureForce + (neighMass * pressOverDens * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, smoothingLength));

//                            accPressureForce = accPressureForce + (neighMass * (thisPressure+neighPressure) / (neighDensity + neighDensity) * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, smoothingLength));


                            accPressureForce = accPressureForce + ( ((thisPressure/(thisDensity*thisDensity)) + (neighPressure/(neighDensity*neighDensity))) * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, particle.smoothingLength) );
                        }
                    }
                }
            }
        }


        if(!accumulate)
        {
//            pressureForce[thisParticleGlobalIdx] = -1.0f * accPressureForce;
            particle.pressureForce[thisParticleGlobalIdx] = -1.0f * thisMass* thisMass * accPressureForce;
        }
        else
        {
            particle.pressureForce[thisParticleGlobalIdx] = particle.pressureForce[thisParticleGlobalIdx] + (-1.0f *thisMass * thisMass * accPressureForce);
//            pressureForce[thisParticleGlobalIdx] = pressureForce[thisParticleGlobalIdx] + (-1.0f * accPressureForce);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputePressureForceFluidRigid_kernel(ParticleGpuData particle, RigidGpuData rigidParticle, const bool accumulate)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particle.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particle.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particle.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {

        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);

        int neighLocalIdx;
        float3 accPressureForce = make_float3(0.0f, 0.0f, 0.0f);


        float thisDensity = particle.den[thisParticleGlobalIdx];
        float thisPressure = particle.pressure[thisParticleGlobalIdx];
        float thisMass = particle.mass;
        float3 thisParticle = particle.pos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = rigidParticle.cellOcc[neighCellIdx];
                    neighCellPartIdx = rigidParticle.cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        float3 neighParticle = rigidParticle.pos[neighParticleGlobalIdx];
                        float neighVolume = rigidParticle.volume[neighParticleGlobalIdx];
                        float pressOverDens = (fabs(thisDensity)<FLT_EPSILON ? 0.0f: (thisPressure) / (thisDensity*thisDensity));

                        accPressureForce = accPressureForce + (thisMass * neighVolume * particle.restDen * pressOverDens * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, particle.smoothingLength));
                    }
                }
            }
        }


        if(!accumulate)
        {
            particle.pressureForce[thisParticleGlobalIdx] = -1.0 * accPressureForce;
//            pressureForce[thisParticleGlobalIdx] = -1.0 * thisMass * thisMass * accPressureForce;
        }
        else
        {
            particle.pressureForce[thisParticleGlobalIdx] = particle.pressureForce[thisParticleGlobalIdx] + (-1.0f * accPressureForce);
//            pressureForce[thisParticleGlobalIdx] = pressureForce[thisParticleGlobalIdx] + (-1.0f * thisMass * thisMass * accPressureForce);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeViscousForce_kernel(FluidGpuData particleData)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;



    if(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);

        int neighLocalIdx;
        float3 accViscForce = make_float3(0.0f, 0.0f, 0.0f);


        float3 thisPos = particleData.pos[thisParticleGlobalIdx];
        float3 thisVel = particleData.vel[thisParticleGlobalIdx];
        float neighMass = particleData.mass;

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = particleData.cellOcc[neighCellIdx];
                    neighCellPartIdx = particleData.cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx == thisParticleGlobalIdx){continue;}

                        float3 neighPos = particleData.pos[neighParticleGlobalIdx];
                        float3 neighVel = particleData.vel[neighParticleGlobalIdx];
                        float neighDensity = particleData.den[neighParticleGlobalIdx];


                        float neighMassOverDen = ( (fabs(neighDensity)<FLT_EPSILON) ? 0.0f : neighMass / neighDensity );

                        accViscForce = accViscForce + ( neighMassOverDen * (neighVel - thisVel) * Poly6Laplacian_Kernel(length(thisPos - neighPos), particleData.smoothingLength) );
                    }
                }
            }
        }

        particleData.viscousForce[thisParticleGlobalIdx] = -1.0f * particleData.viscosity * accViscForce;
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeSurfaceTensionForce_kernel(FluidGpuData particleData)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    __shared__ int s_neighCellOcc[27];
    __shared__ int s_neighCellPartIdx[27];
    __shared__ float3 s_pos[27*27];

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();
    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);

        if(threadIdx.x == 0)
        {
            for(z = zMin; z <= zMax; z++)
            {
                for(y = yMin; y <= yMax; y++)
                {
                    for(x = xMin; x <= xMax; x++)
                    {
                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
                        neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);
                        s_neighCellOcc[nIdx] = particleData.cellOcc[neighCellIdx];
                        s_neighCellPartIdx[nIdx] = particleData.cellPartIdx[neighCellIdx];
                    }
                }
            }
        }
//        if(threadIdx.x < 27)// (1+xMax-xMin)*(1+yMax-yMin)*(1+zMax-zMin))
//        {
//            x = threadIdx.x % 3;
//            y = (threadIdx.x/3) % 3;
//            z = (threadIdx.x/9) % 3;
//            int nIdx = x + (y*3) + (z*9);

//            if( (x-1)<xMin || (x-1)>xMax ||
//                (y-1)<yMin || (y-1)>yMax ||
//                (z-1)<zMin || (z-1)>zMax)
//            {
//                s_neighCellOcc[nIdx] = 0;
//                s_neighCellPartIdx[nIdx] = 0;
//            }
//            else
//            {
//                neighCellIdx = thisCellIdx + (x-1) + ((y-1)*gridDim.x) + ((z-1)*gridDim.x*gridDim.y);

//                s_neighCellOcc[nIdx] = cellOcc[neighCellIdx];
//                s_neighCellPartIdx[nIdx] = cellPartIdx[neighCellIdx];

//            }
//        }
//        __syncthreads();


        int neighLocalIdx;
//        if(threadIdx.x == 0)
//        {
//            neighCellOcc = s_neighCellOcc[threadIdx.x];
//            neighCellPartIdx = s_neighCellPartIdx[threadIdx.x];

//            for(z = zMin; z <= zMax; z++)
//            {
//                for(y = yMin; y <= yMax; y++)
//                {
//                    for(x = xMin; x <= xMax; x++)
//                    {
//                        int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);

//                for(neighLocalIdx=0; neighLocalIdx<neighCellOcc&&neighLocalIdx<27; neighLocalIdx++)
//                {
//                    neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
//                    s_pos[(27*nIdx)+neighLocalIdx] = position[neighParticleGlobalIdx];
//                }
//                    }
//                }
//            }
//        }
        __syncthreads();

        float3 thisPos = particleData.pos[thisParticleGlobalIdx];
        float3 accColourFieldGrad = make_float3(0.0f, 0.0f, 0.0f);
        float accCurvature = 0.0f;
        float neighMass = particleData.mass;

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    int nIdx = (x-xMin) + ((y-yMin)*3) + ((z-zMin)*3*3);
//                    int nIdx = (x+1) + ((y+1)*3) + ((z+1)*3*3);
                    neighCellOcc = s_neighCellOcc[nIdx];
                    neighCellPartIdx = s_neighCellPartIdx[nIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc&&neighLocalIdx<27; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx == thisParticleGlobalIdx){continue;}

                        float3 neighPos = particleData.pos[neighParticleGlobalIdx];
                        float neighDensity = particleData.den[neighParticleGlobalIdx];
//                        float3 neighPos = s_pos[(27*nIdx)+neighLocalIdx];

                        float neighMassOverDen = ( (fabs(neighDensity)<FLT_EPSILON) ? 0.0f : neighMass / neighDensity );

                        accColourFieldGrad = accColourFieldGrad + ( neighMassOverDen * SpikyKernelGradientV_Kernel(thisPos, neighPos, particleData.smoothingLength) );
                        accCurvature = accCurvature + (neighMassOverDen * -Poly6Laplacian_Kernel(length(thisPos - neighPos), particleData.smoothingLength));

                    }
                }
            }
        }

        float colourFieldGradMag = length(accColourFieldGrad);
        if( colourFieldGradMag > particleData.surfaceThreshold )
        {
            accCurvature /= colourFieldGradMag;
            particleData.surfaceTensionForce[thisParticleGlobalIdx] = (particleData.surfaceTension * accCurvature * accColourFieldGrad);
        }
        else
        {
            particleData.surfaceTensionForce[thisParticleGlobalIdx] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeForce_kernel(FluidGpuData particleData, const bool accumulate)
{    
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(!(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
    {
        return;
    }


        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;
        int neighLocalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);


        float thisPressure = particleData.pressure[thisParticleGlobalIdx];
        float3 thisPos = particleData.pos[thisParticleGlobalIdx];
        float3 thisVel = particleData.vel[thisParticleGlobalIdx];
        float3 accPressureForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 accViscForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 accColourFieldGrad = make_float3(0.0f, 0.0f, 0.0f);
        float accCurvature = 0.0f;
        float neighMass = particleData.mass;

        int idx = 0;
        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = particleData.cellOcc[neighCellIdx];
                    neighCellPartIdx = particleData.cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx != thisParticleGlobalIdx)
                        {
                            float3 neighPos = particleData.pos[neighParticleGlobalIdx];
                            float3 neighVel = particleData.vel[neighParticleGlobalIdx];
                            float neighPressure = particleData.pressure[neighParticleGlobalIdx];
                            float neighDensity = particleData.den[neighParticleGlobalIdx];

                            float3 gradW = SpikyKernelGradientV_Kernel(thisPos, neighPos, particleData.smoothingLength);
                            float W = Poly6Laplacian_Kernel(length(thisPos - neighPos), particleData.smoothingLength);

                            float pressOverDens = (fabs(neighDensity)<FLT_EPSILON ? 0.0f: (thisPressure + neighPressure) / (2.0f* neighDensity));
                            accPressureForce = accPressureForce + (neighMass * pressOverDens * gradW);

                            float neighMassOverDen = ( (fabs(neighDensity)<FLT_EPSILON) ? 0.0f : neighMass / neighDensity );
                            accViscForce = accViscForce + ( neighMassOverDen * (neighVel - thisVel) * W );

                            accColourFieldGrad = accColourFieldGrad + ( neighMassOverDen * gradW );
                            accCurvature = accCurvature + (neighMassOverDen * -W);
                        }
                        else
                        {
                            idx++;
                        }
                    }
                }
            }
        }


        accPressureForce = -1.0f * accPressureForce;
        accPressureForce = accumulate ? particleData.pressureForce[thisParticleGlobalIdx] + accPressureForce : accPressureForce;
        particleData.pressureForce[thisParticleGlobalIdx] = accPressureForce;


        accViscForce = -1.0f * particleData.viscosity * accViscForce;
        particleData.viscousForce[thisParticleGlobalIdx] = accViscForce;


        float colourFieldGradMag = length(accColourFieldGrad);
        float3 accSurfTenForce = (colourFieldGradMag > particleData.surfaceThreshold ) ? (-1.0f * particleData.surfaceTension * (accCurvature/colourFieldGradMag) * accColourFieldGrad) : make_float3(0.0f,0.0f,0.0f);
        particleData.surfaceTensionForce[thisParticleGlobalIdx] = accSurfTenForce;
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::ComputeTotalForce_kernel(const bool accumulatePressure,
                                                         const bool accumulateViscous,
                                                         const bool accumulateSurfTen,
                                                         const bool accumulateExternal,
                                                         const bool accumulateGravity,
                                                         FluidGpuData particleData)
{

    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        // re-initialise forces to zero
        float3 accForce = make_float3(0.0f, 0.0f, 0.0f);

        // Add external force
        if(accumulateExternal)
        {
            float3 extForce = particleData.externalForce[thisCellIdx];
            if(isnan(extForce.x) || isnan(extForce.y) || isnan(extForce.z))
            {
//                printf("nan external force\n");
            }
            else
            {
                accForce = accForce + extForce;
            }
        }


        // Add pressure force
        if(accumulatePressure)
        {
            float3 pressForce = particleData.pressureForce[thisParticleGlobalIdx];
            if(isnan(pressForce.x) || isnan(pressForce.y) || isnan(pressForce.z))
            {
//                printf("nan pressure force\n");
            }
            else
            {
                accForce = accForce + pressForce;
            }
        }


        // Add Viscous force
        if(accumulateViscous)
        {
            float3 viscForce = particleData.viscousForce[thisParticleGlobalIdx];
            if(isnan(viscForce.x) || isnan(viscForce.y) || isnan(viscForce.z))
            {
//                printf("nan visc force\n");
            }
            else
            {
                accForce = accForce + viscForce;
            }
        }


        // Add surface tension force
        if(accumulateSurfTen)
        {
            float3 surfTenForce = particleData.surfaceTensionForce[thisParticleGlobalIdx];
            if(isnan(surfTenForce.x) || isnan(surfTenForce.y) || isnan(surfTenForce.z))
            {
//                printf("nan surfTen force\n");
            }
            else
            {
                //printf("%f, %f, %f\n",surfTenForce.x, surfTenForce.y,surfTenForce.z);
                accForce = accForce + surfTenForce;
            }
        }


        // Work out acceleration from force
        float3 acceleration = accForce / particleData.mass;

        // Add gravity acceleration
        if(accumulateGravity)
        {
            acceleration = acceleration + particleData.gravity;
        }

        // Set particle force
        particleData.totalForce[thisParticleGlobalIdx] = acceleration;
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::Integrate_kernel(ParticleGpuData particleData, const float _dt)
{
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < particleData.numParticles)
    {
        //---------------------------------------------------------
        // Good old instable Euler integration - ONLY FOR TESTING
        float3 pos = particleData.pos[idx];
        float3 vel = particleData.vel[idx];
        float3 acc = particleData.totalForce[idx];

        vel = vel + (_dt * acc);
        pos = pos + (_dt * vel);

        //---------------------------------------------------------
        // Verlet/Leapfrog integration
//        float3 pos = particleData.pos[idx];
//        float3 vel = particleData.vel[idx];
//        float3 acc = particleData.totalForce[idx];

//        vel = vel + (0.5f * _dt * acc);
//        pos = pos + (_dt * vel);

//        float pos2 = length2(pos);

//        acc = (-1.0f*pos) / (pos2*sqrtf(pos2));
//        vel = vel + (0.5f * _dt * acc);



        //---------------------------------------------------------
        // TODO:
        // Verlet integration
        // RK4 integration
//        float3 pos = particleData.pos[idx];
//        float3 vel = particleData.vel[idx];
//        float3 acc = particleData.totalForce[idx];

//        float3 newPos = ((2*pos) - vel) + (acc * _dt * _dt);
//        vel = pos;
//        pos = newPos;


        //---------------------------------------------------------
        // Error checking and setting new values

        if(isnan(vel.x) || isnan(vel.y) || isnan(vel.z))
        {
//            printf("nan vel\n");
        }
        else
        {
            particleData.vel[idx] = vel;
        }

        if(isnan(pos.x) || isnan(pos.y) || isnan(pos.z))
        {
//            printf("nan pos\n");
        }
        else
        {
            particleData.pos[idx] = pos;
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::HandleBoundaries_Kernel(ParticleGpuData particleData, const float boundary)
{
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < particleData.numParticles)
    {

        float3 pos = particleData.pos[idx];
        float3 vel = particleData.vel[idx];

        float boundaryDamper = 0.4f;

        if(pos.x < -boundary)
        {
           pos.x = -boundary  + fabs(fabs(pos.x) - boundary);
           vel = make_float3(boundaryDamper*fabs(vel.x),vel.y,vel.z);
        }
        if(pos.x > boundary)
        {
           pos.x = boundary - fabs(fabs(pos.x) - boundary);
           vel = make_float3(-boundaryDamper*fabs(vel.x),vel.y,vel.z);
        }

        if(pos.y < -boundary)
        {
           pos.y = -boundary + fabs(fabs(pos.y) - boundary);
           vel = make_float3(vel.x,boundaryDamper*fabs(vel.y),vel.z);
        }
        if(pos.y > boundary)
        {
           pos.y = boundary - fabs(fabs(pos.y) - boundary);
           vel = make_float3(vel.x,-boundaryDamper*fabs(vel.y),vel.z);
        }

        if(pos.z < -boundary)
        {
           pos.z = -boundary + fabs(fabs(pos.z) - boundary);
           vel = make_float3(vel.x,vel.y,boundaryDamper*fabs(vel.z));
        }
        if(pos.z > boundary)
        {
           pos.z = boundary - fabs(fabs(pos.z) - boundary);
           vel = make_float3(vel.x,vel.y,-boundaryDamper*fabs(vel.z));
        }

        particleData.pos[idx] = pos;
        particleData.vel[idx] = vel;
    }
}

//--------------------------------------------------------------------------------------------------------------------

__global__ void sphGPU_Kernels::InitParticleAsCube_Kernel(ParticleGpuData particle,
                                                          const uint numPartsPerAxis,
                                                          const float scale)
{

    uint x = threadIdx.x + (blockIdx.x * blockDim.x);
    uint y = threadIdx.y + (blockIdx.y * blockDim.y);
    uint z = threadIdx.z + (blockIdx.z * blockDim.z);
    uint idx = x + (y * numPartsPerAxis) + (z * numPartsPerAxis * numPartsPerAxis);

    if(x >= numPartsPerAxis || y >= numPartsPerAxis || z >= numPartsPerAxis || idx >= particle.numParticles)
    {
        return;
    }

    float posX = scale * (x - (0.5f * numPartsPerAxis));
    float posY = scale * (y - (0.5f * numPartsPerAxis));
    float posZ = scale * (z - (0.5f * numPartsPerAxis));

    particle.pos[idx] = make_float3(posX, posY, posZ);
    particle.vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
    particle.den[idx] = particle.restDen;
}



//--------------------------------------------------------------------------------------------------------------------
// Algae functions
__global__ void sphGPU_Kernels::ComputeAdvectionForce(ParticleGpuData particleData, FluidGpuData advectorParticleData, const bool accumulate)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;
    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);

        int neighLocalIdx;


        float3 thisPos = particleData.pos[thisParticleGlobalIdx];
        float3 accForce = make_float3(0.0f, 0.0f, 0.0f);
//        vel[thisParticleGlobalIdx] = vel[thisParticleGlobalIdx]*0.9f;//make_float3(0.0f, 0.0f, 0.0f);

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = advectorParticleData.cellOcc[neighCellIdx];
                    neighCellPartIdx = advectorParticleData.cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        float3 neighPos = advectorParticleData.pos[neighParticleGlobalIdx];

                        float W = Poly6Kernel_Kernel(length(thisPos-neighPos), particleData.smoothingLength);
                        float invDensity = 1.0f / advectorParticleData.den[neighParticleGlobalIdx];

                        accForce = accForce + (advectorParticleData.totalForce[neighParticleGlobalIdx] * W * invDensity);

//                        accForce = accForce + ((neighPos - thisPos) *0.1f* W);
                    }
                }
            }
        }

        accForce = (accForce * advectorParticleData.mass * 1.00f);// + make_float3(0.0f, -0.8f, 0.0f);

        particleData.totalForce[thisParticleGlobalIdx] = accForce;
    }
}

//--------------------------------------------------------------------------------------------------------------------
__global__ void sphGPU_Kernels::AdvectParticle(ParticleGpuData particleData, FluidGpuData advectorParticleData, const float deltaTime)
{
    __shared__ int thisCellIdx;
    __shared__ int thisCellOcc;

    if(threadIdx.x==0)
    {
        thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
        thisCellOcc = particleData.cellOcc[thisCellIdx];
    }
    __syncthreads();

    int thisParticleGlobalIdx = particleData.cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < particleData.numParticles && threadIdx.x < thisCellOcc && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        int neighCellIdx;
        int neighCellOcc;
        int neighCellPartIdx;
        int neighParticleGlobalIdx;

        int x, y, z;
        int xMin = ((blockIdx.x==0)?0:-1);
        int yMin = ((blockIdx.y==0)?0:-1);
        int zMin = ((blockIdx.z==0)?0:-1);
        int xMax = ((blockIdx.x==gridDim.x-1)?0:1);
        int yMax = ((blockIdx.y==gridDim.y-1)?0:1);
        int zMax = ((blockIdx.z==gridDim.z-1)?0:1);

        int neighLocalIdx;


        float3 thisPos = particleData.pos[thisParticleGlobalIdx];
        float3 accVel = make_float3(0.0f, 0.0f, 0.0f);

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = advectorParticleData.cellOcc[neighCellIdx];
                    neighCellPartIdx = advectorParticleData.cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        float3 neighPos = advectorParticleData.pos[neighParticleGlobalIdx];
                        float3 neighVel = advectorParticleData.vel[neighParticleGlobalIdx];

                        float W = Poly6Kernel_Kernel(length(thisPos-neighPos), particleData.smoothingLength);
                        float invDensity = 1.0f / advectorParticleData.den[neighParticleGlobalIdx];
                        accVel = accVel + (neighVel * W * invDensity);
//                        accVel = accVel + ((neighPos - thisPos) *0.1f* W);
                    }
                }
            }
        }

        particleData.vel[thisParticleGlobalIdx] = (particleData.vel[thisParticleGlobalIdx]*0.5f) + (accVel * advectorParticleData.mass * 0.50f);
        particleData.pos[thisParticleGlobalIdx] = thisPos + (accVel * deltaTime);
    }
}

//--------------------------------------------------------------------------------------------------------------------
__global__ void sphGPU_Kernels::ComputeBioluminescence(AlgaeGpuData particleData, const float deltaTime)
{
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < particleData.numParticles)
    {
        float currIllum = particleData.illum[idx];
        float press = particleData.pressure[idx];
        float prevPress = particleData.prevPressure[idx];
        particleData.prevPressure[idx] = press;

        float deltaPress = /*fabs*/(press - prevPress);
        float deltaIllum = ((deltaPress > particleData.bioluminescenceThreshold) ? particleData.reactionRate : -particleData.deactionRate);

        const float maxIllum = 1.0f;
        const float minIllum = 0.0f;

        currIllum += (deltaIllum * deltaTime);
        currIllum = (currIllum < minIllum) ? minIllum : currIllum;
        currIllum = (currIllum > maxIllum) ? maxIllum : currIllum;

        particleData.illum[idx] = (isnan(currIllum) ? minIllum : currIllum);

    }

}
