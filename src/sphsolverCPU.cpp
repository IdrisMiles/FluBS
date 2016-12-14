#include "include/sphsolverCPU.h"

SPHSolverCPU::SPHSolverCPU(FluidProperty *_fluidProperty)
{

}

SPHSolverCPU::~SPHSolverCPU()
{

}


void SPHSolverCPU::Init()
{

}

void SPHSolverCPU::Solve(float _dt, glm::vec3 *_d_p, glm::vec3 *_d_v)
{

}

void SPHSolverCPU::ParticleHash(unsigned int *hash, unsigned int *cellOcc, glm::vec3 *particles, const unsigned int N, const unsigned int gridRes, const float cellWidth)
{
    for(unsigned int i=0; i<m_fluidProperty->numParticles; i++)
    {
        float gridDim = gridRes * cellWidth;
        float invGridDim = 1.0f / gridDim;
        glm::vec3 particle = particles[i];
        uint hashID;

        // Get normalised particle positions [0-1]
        float normX = (particle.x + (0.5f * gridDim)) * invGridDim;
        float normY = (particle.y + (0.5f * gridDim)) * invGridDim;
        float normZ = (particle.z + (0.5f * gridDim)) * invGridDim;

        if(normX > 1.0f || normX < 0.0f ||
           normY > 1.0f || normY < 0.0f ||
           normZ > 1.0f || normZ < 0.0f )
        {
            hashID = 0;
        }
        else
        {
            // Get hash values for x, y, z
            uint hashX = floor(normX * gridRes);
            uint hashY = floor(normY * gridRes);
            uint hashZ = floor(normZ * gridRes);

            hashID = hashX + (hashY * gridRes) + (hashZ * gridRes * gridRes);
        }

        // Update hash id for this particle
        hash[i] = hashID;


        // Update cell occupancy for the cell
        cellOcc[hashID]++;
    }
}


void SPHSolverCPU::ComputePressure(float *pressure, float *density, const float restDensity, const uint *cellOcc, const uint *cellPartIdx, const glm::vec3 *particles, const uint numPoints, const float smoothingLength)
{

}

void SPHSolverCPU::ComputePressureForce(const uint maxCellOcc, glm::vec3 *pressureForce, const float *pressure, const float *density, const float *mass, const glm::vec3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{

}

void SPHSolverCPU::ComputeTotalForce(const uint maxCellOcc, glm::vec3 *force, const glm::vec3 *externalForce, const glm::vec3 *pressureForce, const float *mass, const glm::vec3 *particles, const glm::vec3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength)
{

}

void SPHSolverCPU::Integrate(const uint maxCellOcc, glm::vec3 *force, glm::vec3 *particles, glm::vec3 *velocities, const float _dt, const uint numPoints)
{

}


void SPHSolverCPU::InitParticleAsCube(glm::vec3 *particles, glm::vec3 *velocities, const unsigned int numParticles, const unsigned int numPartsPerAxis, const float scale)
{

}
