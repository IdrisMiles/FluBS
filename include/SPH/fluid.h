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

    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual FluidProperty *GetProperty();

    void MapCudaGLResources();
    void ReleaseCudaGLResources();

    float3 *GetViscForcePtr();
    void ReleaseViscForcePtr();

    float3 *GetSurfTenForcePtr();
    void ReleaseSurfTenForcePtr();

    float3 *GetPredictPosPtr();
    void ReleasePredictPosPtr();

    float3 *GetPredictVelPtr();
    void ReleasePredictVelPtr();

    float *GetDensityErrPtr();
    void ReleaseDensityErrPtr();


protected:
    void InitFluidAsMesh();
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpGL();
    virtual void CleanUpCUDAMemory();


    // specfic simulation Data
    std::shared_ptr<FluidProperty> m_fluidProperty;
    float3* d_viscousForcePtr;
    float3* d_surfaceTensionForcePtr;

    float3 *d_predictPositionPtr;
    float3 *d_predictVelocityPtr;
    float *d_densityErrPtr;


};

#endif // FLUID_H
