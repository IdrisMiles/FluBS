#ifndef FLUID_H
#define FLUID_H

#include "SPH/fluidproperty.h"
#include "SPH/isphparticles.h"
#include "FluidSystem/fluidsolverproperty.h"



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

protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void CleanUpCUDAMemory();
    void InitFluidAsMesh();


    // Simulation Data
    std::shared_ptr<FluidProperty> m_fluidProperty;
    float3* d_viscousForcePtr;
    float3* d_surfaceTensionForcePtr;


    virtual void InitGL();
    virtual void InitVAO();
    virtual void CleanUpGL();

};

#endif // FLUID_H
