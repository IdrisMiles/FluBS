#ifndef RIGID_H
#define RIGID_H

// sph includes
#include "SPH/isphparticles.h"
#include "SPH/rigidproperty.h"



class Rigid : public BaseSphParticle
{
public:
    Rigid(std::shared_ptr<RigidProperty> _rigidProperty, Mesh _mesh);
    virtual ~Rigid();



    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual RigidProperty* GetProperty();

    virtual void MapCudaGLResources();
    virtual void ReleaseCudaGLResources();


    float *GetVolumePtr();
    void ReleaseVolumePtr();


protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();

    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();


    // Simulation Data
    std::shared_ptr<RigidProperty> m_property;
    float* d_volumePtr;

};

#endif // RIGID_H
