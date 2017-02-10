#include "../cuda_inc/sphGPU_Kernels.cuh"


#include "../cuda_inc/vec_ops.cuh"
#include "../cuda_inc/smoothingKernel.cuh"


#include <math_constants.h>
#include <stdio.h>
#include <math.h>
#include <float.h>



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

    hashID = hashX + (hashY * gridRes) + (hashZ * gridRes * gridRes);

    if(hashID >= gridRes * gridRes * gridRes)
    {
        printf("Hash out of bounds\n");
        printf("%u, %u, %u\n", hashX, hashY, hashZ);
    }

    // Update hash id for this particle
    hash[idx] = hashID;


    // Update cell occupancy for the cell
    atomicAdd(&cellOcc[hashID], 1u);


}

__global__ void sphGPU_Kernels::ComputeVolume_kernel(float *volume,
                                     const uint *cellOcc,
                                     const uint *cellPartIdx,
                                     const float3 *particles,
                                     const uint numPoints,
                                     const float smoothingLength)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;

    if(thisParticleGlobalIdx < numPoints && threadIdx.x < cellOcc[thisCellIdx] && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
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
        float accVolume = 0.0f;
        float3 thisParticle = particles[thisParticleGlobalIdx];

        uint numNeighCells = 0;
        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    numNeighCells++;
                    neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);

                    neighCellOcc = cellOcc[neighCellIdx];
                    neighCellPartIdx = cellPartIdx[neighCellIdx];
                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        float3 neighParticle = particles[neighParticleGlobalIdx];

                        accVolume += fabs(Poly6Kernel_Kernel(length(thisParticle - neighParticle), smoothingLength));
                    }
                }
            }
        }

        if(isnan(accVolume) || fabs(accVolume) < FLT_EPSILON)
        {
            volume[thisParticleGlobalIdx] = 1.0f;
        }
        else
        {
            volume[thisParticleGlobalIdx] = accVolume;
        }
    }
}

__global__ void sphGPU_Kernels::ComputeDensity_kernel(float *density,
                                                      const float *mass,
                                                      const uint *cellOcc,
                                                      const uint *cellPartIdx,
                                                      const float3 *particles,
                                                      const uint numPoints,
                                                      const float smoothingLength,
                                                      const bool accumulate)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;



    if((thisParticleGlobalIdx < numPoints) && (threadIdx.x < cellOcc[thisCellIdx]) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
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
        float accDensity = 0.0f;
        float thisDensity = 0.0f;
        float3 thisParticle = particles[thisParticleGlobalIdx];

        unsigned int numNeighs = 0;
        uint numNeighCells = 0;
        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    numNeighCells++;
                    neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);

                    // Get density contribution from other fluid particles
                    neighCellOcc = cellOcc[neighCellIdx];
                    neighCellPartIdx = cellPartIdx[neighCellIdx];
                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighParticle = particles[neighParticleGlobalIdx];

                        thisDensity = mass[neighParticleGlobalIdx] * fabs(Poly6Kernel_Kernel(length(thisParticle - neighParticle), smoothingLength));

                        accDensity += thisDensity;

                        numNeighs++;
                    }
                }
            }
        }

        if(isnan(accDensity))
        {
            printf("nan density \n");

            if(!accumulate)
            {
                density[thisParticleGlobalIdx] = 0.0f;
            }
        }
        else
        {
            if(accumulate)
            {
                atomicAdd(&density[thisParticleGlobalIdx], accDensity);
            }
            else
            {
                density[thisParticleGlobalIdx] = accDensity;
            }
        }


    }

}



__global__ void sphGPU_Kernels::ComputeDensityFluidRigid_kernel(const uint numPoints,
                                                                const float fluidRestDensity,
                                                                float *fluidDensity,
                                                                const uint *fluidCellOcc,
                                                                const uint *fluidCellPartIdx,
                                                                const float3 *fluidPos,
                                                                float *rigidVolume,
                                                                const uint *rigidCellOcc,
                                                                const uint *rigidCellPartIdx,
                                                                const float3 *rigidPos,
                                                                const float smoothingLength,
                                                                const bool accumulate)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = fluidCellPartIdx[thisCellIdx] + threadIdx.x;



    if((thisParticleGlobalIdx < numPoints) && (threadIdx.x < fluidCellOcc[thisCellIdx]) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
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
        float accDensity = 0.0f;
        float thisDensity = 0.0f;
        float3 thisParticle = fluidPos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);

                    // Get density contribution from other fluid particles
                    neighCellOcc = rigidCellOcc[neighCellIdx];
                    neighCellPartIdx = rigidCellPartIdx[neighCellIdx];
                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighParticle = rigidPos[neighParticleGlobalIdx];

                        thisDensity = fluidRestDensity * rigidVolume[neighParticleGlobalIdx] * fabs(Poly6Kernel_Kernel(length(thisParticle - neighParticle), smoothingLength));

                        accDensity += thisDensity;
                    }
                }
            }
        }

        if(isnan(accDensity))
        {
            printf("nan density \n");

            if(!accumulate)
            {
                fluidDensity[thisParticleGlobalIdx] = 0.0f;
            }
        }
        else
        {
            if(accumulate)
            {
                atomicAdd(&fluidDensity[thisParticleGlobalIdx], accDensity);
            }
            else
            {
                fluidDensity[thisParticleGlobalIdx] = accDensity;
            }
        }

    } // end if valid point
}



__global__ void sphGPU_Kernels::ComputeDensityFluidFluid_kernel(const uint numPoints,
                                                                float *fluidDensity,
                                                                const uint *fluidCellOcc,
                                                                const uint *fluidCellPartIdx,
                                                                const float3 *fluidPos,
                                                                const uint *otherFluidCellOcc,
                                                                const uint *otherFluidCellPartIdx,
                                                                float *otherFluidMass,
                                                                const float3 *otherFluidPos,
                                                                const float smoothingLength,
                                                                const bool accumulate)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = fluidCellPartIdx[thisCellIdx] + threadIdx.x;



    if((thisParticleGlobalIdx < numPoints) && (threadIdx.x < fluidCellOcc[thisCellIdx]) && (thisCellIdx < gridDim.x * gridDim.y * gridDim.z))
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
        float accDensity = 0.0f;
        float thisDensity = 0.0f;
        float3 thisParticle = fluidPos[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {
                    neighCellIdx = thisCellIdx + x + (y*gridDim.x) + (z*gridDim.x*gridDim.y);

                    // Get density contribution from other fluid particles
                    neighCellOcc = otherFluidCellOcc[neighCellIdx];
                    neighCellPartIdx = otherFluidCellPartIdx[neighCellIdx];
                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;

                        float3 neighParticle = otherFluidPos[neighParticleGlobalIdx];

                        thisDensity = otherFluidMass[neighParticleGlobalIdx] * fabs(Poly6Kernel_Kernel(length(thisParticle - neighParticle), smoothingLength));

                        accDensity += thisDensity;
                    }
                }
            }
        }

        if(isnan(accDensity))
        {
            printf("nan density \n");

            if(!accumulate)
            {
                fluidDensity[thisParticleGlobalIdx] = 0.0f;
            }
        }
        else
        {
            if(accumulate)
            {
                atomicAdd(&fluidDensity[thisParticleGlobalIdx], accDensity);
            }
            else
            {
                fluidDensity[thisParticleGlobalIdx] = accDensity;
            }
        }

    } // end if valid point
}






__global__ void sphGPU_Kernels::ComputePressure_kernel(float *pressure,
                                                       float *density,
                                                       const float restDensity,
                                                       const float gasConstant,
                                                       const uint *cellOcc,
                                                       const uint *cellPartIdx,
                                                       const uint numPoints)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;



    if(thisParticleGlobalIdx < numPoints && threadIdx.x < cellOcc[thisCellIdx] && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        //float beta = 0.35;
        //float gamma = 7.0f;
        //float accPressure = beta * (pow((accDensity/restDensity), gamma)-1.0f);
        //float accPressure = gasConstant * ((accDensity/restDensity) - 1.0f);

        float accPressure = gasConstant * (density[thisParticleGlobalIdx] - restDensity);

        if(isnan(accPressure))
        {
            printf("nan pressure \n");
            pressure[thisParticleGlobalIdx] = 0.0f;
        }
        else
        {
            pressure[thisParticleGlobalIdx] = accPressure;
        }

    }

}

__global__ void sphGPU_Kernels::ComputePressureForce_kernel(float3 *pressureForce,
                                                            const float *pressure,
                                                            const float *density,
                                                            const float *mass,
                                                            const float3 *particles,
                                                            const uint *cellOcc,
                                                            const uint *cellPartIdx,
                                                            const uint numPoints,
                                                            const float smoothingLength,
                                                            const bool accumulate)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < numPoints && threadIdx.x < cellOcc[thisCellIdx] && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
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


        float thisPressure = pressure[thisParticleGlobalIdx];
        float3 thisParticle = particles[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = cellOcc[neighCellIdx];
                    neighCellPartIdx = cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx != thisParticleGlobalIdx)
                        {
                            float3 neighParticle = particles[neighParticleGlobalIdx];
                            float neighPressure = pressure[neighParticleGlobalIdx];
                            float neighDensity = density[neighParticleGlobalIdx];
                            float neighMass = mass[neighParticleGlobalIdx];

                            float pressOverDens = (fabs(neighDensity)<FLT_EPSILON ? 0.0f: (thisPressure + neighPressure) / (2.0f* neighDensity));

                            accPressureForce = accPressureForce + (neighMass * pressOverDens * SpikyKernelGradientV_Kernel(thisParticle, neighParticle, smoothingLength));
                        }
                    }
                }
            }
        }


        if(accumulate)
        {
            pressureForce[thisParticleGlobalIdx] = -1.0f * accPressureForce;
        }
        else
        {
            pressureForce[thisParticleGlobalIdx] = pressureForce[thisParticleGlobalIdx] + (-1.0f * accPressureForce);
        }
    }
}

__global__ void sphGPU_Kernels::ComputeViscousForce_kernel(float3 *viscForce,
                                                           const float viscCoeff,
                                                           const float3 *velocity,
                                                           const float *density,
                                                           const float *mass,
                                                           const float3 *position,
                                                           const uint *cellOcc,
                                                           const uint *cellPartIdx,
                                                           const uint numPoints,
                                                           const float smoothingLength)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < numPoints && threadIdx.x < cellOcc[thisCellIdx] && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
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


        float3 thisPos = position[thisParticleGlobalIdx];
        float3 thisVel = velocity[thisParticleGlobalIdx];

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = cellOcc[neighCellIdx];
                    neighCellPartIdx = cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx == thisParticleGlobalIdx){continue;}

                        float3 neighPos = position[neighParticleGlobalIdx];
                        float3 neighVel = velocity[neighParticleGlobalIdx];
                        float neighDensity = density[neighParticleGlobalIdx];
                        float neighMass = mass[neighParticleGlobalIdx];
                        float neighMassOverDen = ( (fabs(neighDensity)<FLT_EPSILON) ? 0.0f : neighMass / neighDensity );

                        accViscForce = accViscForce + ( neighMassOverDen * (neighVel - thisVel) * Poly6Laplacian_Kernel(length(thisPos - neighPos), smoothingLength) );
                    }
                }
            }
        }

        viscForce[thisParticleGlobalIdx] = -1.0f * viscCoeff * accViscForce;
    }
}


__global__ void sphGPU_Kernels::ComputeSurfaceTensionForce_kernel(float3 *surfaceTensionForce,
                                                                  const float surfaceTension,
                                                                  const float surfaceThreshold,
                                                                  /*const*/ float *density,
                                                                  const float *mass,
                                                                  const float3 *position,
                                                                  const uint *cellOcc,
                                                                  const uint *cellPartIdx,
                                                                  const uint numPoints,
                                                                  const float smoothingLength)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;


    if(thisParticleGlobalIdx < numPoints && threadIdx.x < cellOcc[thisCellIdx] && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
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


        float3 thisPos = position[thisParticleGlobalIdx];
        float3 accColourFieldGrad = make_float3(0.0f, 0.0f, 0.0f);
        float accCurvature = 0.0f;

        for(z = zMin; z <= zMax; z++)
        {
            for(y = yMin; y <= yMax; y++)
            {
                for(x = xMin; x <= xMax; x++)
                {

                    neighCellIdx = (blockIdx.x + x) + ((blockIdx.y + y) * gridDim.x) + ((blockIdx.z + z) * gridDim.x * gridDim.y);
                    neighCellOcc = cellOcc[neighCellIdx];
                    neighCellPartIdx = cellPartIdx[neighCellIdx];

                    for(neighLocalIdx=0; neighLocalIdx<neighCellOcc; neighLocalIdx++)
                    {
                        neighParticleGlobalIdx = neighCellPartIdx + neighLocalIdx;
                        if(neighParticleGlobalIdx == thisParticleGlobalIdx){continue;}

                        float3 neighPos = position[neighParticleGlobalIdx];
                        float neighDensity = density[neighParticleGlobalIdx];
                        float neighMass = mass[neighParticleGlobalIdx];
                        float neighMassOverDen = ( (fabs(neighDensity)<FLT_EPSILON) ? 0.0f : neighMass / neighDensity );

                        accColourFieldGrad = accColourFieldGrad + ( neighMassOverDen * SpikyKernelGradientV_Kernel(thisPos, neighPos, smoothingLength) );
                        accCurvature = accCurvature + (neighMassOverDen * -Poly6Laplacian_Kernel(length(thisPos - neighPos), smoothingLength));

                    }
                }
            }
        }

        float colourFieldGradMag = length(accColourFieldGrad);
        if( colourFieldGradMag > surfaceThreshold )
        {
            accCurvature /= colourFieldGradMag;
            surfaceTensionForce[thisParticleGlobalIdx] = (-surfaceTension * accCurvature * accColourFieldGrad);
        }
        else
        {
            surfaceTensionForce[thisParticleGlobalIdx] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

__global__ void sphGPU_Kernels::ComputeForces_kernel(float3 *force,
                                                     const float3 *externalForce,
                                                     const float3 *pressureForce,
                                                     const float3 *viscousForce,
                                                     const float3 *surfaceTensionForce,
                                                     const float *mass,
                                                     const float3 *particles,
                                                     const float3 *velocities,
                                                     const uint *cellOcc,
                                                     const uint *cellPartIdx,
                                                     const uint numPoints,
                                                     const float smoothingLength)
{
    int thisCellIdx = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int thisParticleGlobalIdx = cellPartIdx[thisCellIdx] + threadIdx.x;

    if(thisParticleGlobalIdx < numPoints && threadIdx.x < cellOcc[thisCellIdx] && thisCellIdx < gridDim.x * gridDim.y * gridDim.z)
    {
        // re-initialise forces to zero
        float3 accForce = make_float3(0.0f, 0.0f, 0.0f);

        // Add external force
        float3 extForce = externalForce[thisCellIdx];
        if(isnan(extForce.x) || isnan(extForce.y) || isnan(extForce.z))
        {
            printf("nan external force\n");
        }
        else
        {
            accForce = accForce + extForce;
        }


        // Add pressure force
        float3 pressForce = pressureForce[thisParticleGlobalIdx];
        if(isnan(pressForce.x) || isnan(pressForce.y) || isnan(pressForce.z))
        {
            printf("nan pressure force\n");
        }
        else
        {
            accForce = accForce + pressForce;
        }

        // Add Viscous force
        float3 viscForce = viscousForce[thisParticleGlobalIdx];
        if(isnan(viscForce.x) || isnan(viscForce.y) || isnan(viscForce.z))
        {
            printf("nan visc force\n");
        }
        else
        {
            accForce = accForce + viscForce;
        }

        // Add surface tension force
        float3 surfTenForce = surfaceTensionForce[thisParticleGlobalIdx];
        if(isnan(surfTenForce.x) || isnan(surfTenForce.y) || isnan(surfTenForce.z))
        {
            printf("nan surfTen force\n");
        }
        else
        {
            //printf("%f, %f, %f\n",surfTenForce.x, surfTenForce.y,surfTenForce.z);
            accForce = accForce + surfTenForce;
        }


        // Work out acceleration from force
        float3 acceleration = accForce / mass[thisParticleGlobalIdx];

        // Add gravity acceleration
        acceleration = acceleration + make_float3(0.0f, -9.81f, 0.0f);

        // Set particle force
        force[thisParticleGlobalIdx] = acceleration;
    }
}

__global__ void sphGPU_Kernels::Integrate_kernel(float3 *force,
                                                 float3 *particles,
                                                 float3 *velocities,
                                                 const float _dt,
                                                 const uint numPoints)
{
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < numPoints)
    {
        //---------------------------------------------------------
        // Good old instable Euler integration - ONLY FOR TESTING
        float3 oldPos = particles[idx];
        float3 oldVel = velocities[idx];

        float3 newVel = oldVel + (_dt * force[idx]);
        float3 newPos = oldPos + (_dt * newVel);

        //---------------------------------------------------------
        // Verlet/Leapfrog integration
//        float3 newPos = oldPos + (oldVel * _dt) + (0.5f * force[idx] * _dt * _dt);
//        float3 newVel = oldVel + (0.5 * (force[idx] + force[idx]) * _dt);

        //---------------------------------------------------------
        // TODO:
        // Verlet integration
        // RK4 integration

        //---------------------------------------------------------
        // Error checking and setting new values

        if(isnan(newVel.x) || isnan(newVel.y) || isnan(newVel.z))
        {
            printf("nan vel\n");
        }
        else
        {
            velocities[idx] = newVel;
        }

        if(isnan(newPos.x) || isnan(newPos.y) || isnan(newPos.z))
        {
            printf("nan pos\n");
        }
        else
        {
            particles[idx] = newPos;
        }
    }
}

__global__ void sphGPU_Kernels::HandleBoundaries_Kernel(float3 *particles,
                                                        float3 *velocities,
                                                        const float boundary,
                                                        const uint numPoints)
{
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < numPoints)
    {

        float3 pos = particles[idx];
        float3 vel = velocities[idx];

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

        particles[idx] = pos;
        velocities[idx] = vel;
    }
}

__global__ void sphGPU_Kernels::InitParticleAsCube_Kernel(float3 *particles,
                                                          float3 *velocities,
                                                          float *densities,
                                                          const float restDensity,
                                                          const uint numParticles,
                                                          const uint numPartsPerAxis,
                                                          const float scale)
{

    uint x = threadIdx.x + (blockIdx.x * blockDim.x);
    uint y = threadIdx.y + (blockIdx.y * blockDim.y);
    uint z = threadIdx.z + (blockIdx.z * blockDim.z);
    uint idx = x + (y * numPartsPerAxis) + (z * numPartsPerAxis * numPartsPerAxis);

    if(x >= numPartsPerAxis || y >= numPartsPerAxis || z >= numPartsPerAxis || idx >= numParticles)
    {
        return;
    }

    float posX = scale * (x - (0.5f * numPartsPerAxis));
    float posY = scale * (y - (0.5f * numPartsPerAxis));
    float posZ = scale * (z - (0.5f * numPartsPerAxis));

    particles[idx] = make_float3(posX, posY, posZ);
    velocities[idx] = make_float3(0.0f, 0.0f, 0.0f);
    densities[idx] = restDensity;
}



