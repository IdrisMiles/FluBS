#ifndef FLUID_H
#define FLUID_H

// sph includes
#include "SPH/fluidproperty.h"
#include "SPH/isphparticles.h"



class Fluid : public BaseSphParticle
{

public:
    Fluid(std::shared_ptr<FluidProperty> _fluidProperty);
    Fluid(std::shared_ptr<FluidProperty> _rigidProperty, Mesh _mesh);
    virtual ~Fluid();

    virtual void SetupSolveSpecs(const FluidSolverProperty &_solverProps);

    virtual FluidProperty *GetProperty();

    void MapCudaGLResources();
    void ReleaseCudaGLResources();

    float3 *GetViscForcePtr();

    float3 *GetSurfTenForcePtr();

    float3 *GetPredictPosPtr();

    float3 *GetPredictVelPtr();

    float *GetDensityErrPtr();


    virtual void GetPositions(std::vector<glm::vec3> &_pos);
    virtual void GetVelocities(std::vector<glm::vec3> &_vel);
    virtual void GetParticleIds(std::vector<int> &_ids);

    virtual void SetPositions(const std::vector<glm::vec3> &_pos);
    virtual void SetVelocities(const std::vector<glm::vec3> &_vel);
    virtual void SetParticleIds(const std::vector<int> &_ids);


protected:
    void InitFluidAsMesh();
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpGL();
    virtual void CleanUpCUDAMemory();


    // specfic simulation Data
    std::shared_ptr<FluidProperty> m_property;
    float3* d_viscousForcePtr;
    float3* d_surfaceTensionForcePtr;

    float3 *d_predictPositionPtr;
    float3 *d_predictVelocityPtr;
    float *d_densityErrPtr;


    // PCI SPH vars
    // https://graphics.ethz.ch/~sobarbar/papers/Sol09/Sol09.pdf
    // Euqation 8.
    float m_pciBeta; //dt^2 * m^2 * (2/restDensity^2)
    float m_pciSigma;


};

#endif // FLUID_H
