#ifndef ABSTRACTSPHSOLVER_H
#define ABSTRACTSPHSOLVER_H

#include "fluidproperty.h"
#include <glm/glm.hpp>
#include <vector>


class AbstractSPHSolver
{
public:
    AbstractSPHSolver();
    virtual ~AbstractSPHSolver();


    virtual void Init(float *p_pos, float *p_vel) = 0;
//    void Solve(float _dt, float3 *_d_p, float3 *_d_v);

//    void ParticleHash(unsigned int *hash, unsigned int *cellOcc, float3 *particles, const unsigned int N, const unsigned int gridRes, const float cellWidth);
//    void ComputePressure(const uint maxCellOcc, float *pressure, float *density, const float restDensity, const uint *cellOcc, const uint *cellPartIdx, const float3 *particles, const uint numPoints, const float smoothingLength);
//    void ComputePressureForce(const uint maxCellOcc, float3 *pressureForce, const float *pressure, const float *density, const float *mass, const float3 *particles, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
//    void ComputeTotalForce(const uint maxCellOcc, float3 *force, const float3 *externalForce, const float3 *pressureForce, const float *mass, const float3 *particles, const float3 *velocities, const uint *cellOcc, const uint *cellPartIdx, const uint numPoints, const float smoothingLength);
//    void Integrate(const uint maxCellOcc, float3 *force, float3 *particles, float3 *velocities, const float _dt, const uint numPoints);

//    void SPHSolve(const unsigned int maxCellOcc, const unsigned int *cellOcc, const unsigned int *cellIds, float3 *particles, float3 *velocities, const unsigned int numPoints, const unsigned int gridRes, const float smoothingLength, const float dt);

//    void InitParticleAsCube(float3 *particles, float3 *velocities, const unsigned int numParticles, const unsigned int numPartsPerAxis, const float scale);



protected:

    FluidProperty* m_fluidProperty;

};

#endif // ABSTRACTSPHSOLVER_H
