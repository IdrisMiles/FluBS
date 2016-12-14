#ifndef SPHSOLVERCPU_H
#define SPHSOLVERCPU_H

#include "fluidproperty.h"
#include <glm/glm.hpp>
#include <vector>

class SPHSolverCPU
{
public:
    SPHSolverCPU(FluidProperty* _fluidProperty);
    virtual ~SPHSolverCPU();

    void Init();
    void Solve(float _dt, glm::vec3 *_d_p, glm::vec3 *_d_v);

    void ParticleHash(unsigned int *hash, unsigned int *cellOcc, glm::vec3 *particles, const unsigned int N, const unsigned int gridRes, const float cellWidth);
    void ComputePressure(float *pressure, float *density, const float restDensity, const uint *cellOcc, const uint *cellPartIdx, const glm::vec3 *particles, const uint numPoints, const float smoothingLength);
    void ComputePressureForce(const uint maxCellOcc, glm::vec3 *pressureForce, const float *pressure, const float *density, const float *mass, const glm::vec3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
    void ComputeTotalForce(const uint maxCellOcc, glm::vec3 *force, const glm::vec3 *externalForce, const glm::vec3 *pressureForce, const float *mass, const glm::vec3 *particles, const glm::vec3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
    void Integrate(const uint maxCellOcc, glm::vec3 *force, glm::vec3 *particles, glm::vec3 *velocities, const float _dt, const uint numPoints);

    void SPHSolve(const unsigned int maxCellOcc, const unsigned int *cellOcc, const unsigned int *cellIds, glm::vec3 *particles, glm::vec3 *velocities, const unsigned int numPoints, const unsigned int gridRes, const float smoothingLength, const float dt);

    void InitParticleAsCube(glm::vec3 *particles, glm::vec3 *velocities, const unsigned int numParticles, const unsigned int numPartsPerAxis, const float scale);



private:

    FluidProperty* m_fluidProperty;

    std::vector<glm::vec3> d_positions;
    std::vector<glm::vec3> d_velocities;
    std::vector<glm::vec3> d_pressureForces;
    std::vector<glm::vec3> d_externalForces;
    std::vector<glm::vec3> d_totalForces;
    std::vector<float> d_densities;
    std::vector<float> d_pressures;
    std::vector<float> d_mass;
    std::vector<unsigned int> d_particleHashIds;
    std::vector<unsigned int> d_cellOccupancy;
    std::vector<unsigned int> d_cellParticleIdx;  // holds indexes into d_particles for start of each cell

    glm::vec3* d_positions_ptr;
    glm::vec3* d_velocities_ptr;
    glm::vec3* d_pressureForces_ptr;
    glm::vec3* d_externalForces_ptr;
    glm::vec3* d_totalForces_ptr;
    float* d_densities_ptr;
    float* d_pressures_ptr;
    float* d_mass_ptr;
    unsigned int* d_particleHashIds_ptr;
    unsigned int* d_cellOccupancy_ptr;
    unsigned int* d_cellParticleIdx_ptr;
};

#endif // SPHSOLVERCPU_H
