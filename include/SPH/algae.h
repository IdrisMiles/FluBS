#ifndef ALGAE_H
#define ALGAE_H

// sph includes
#include "SPH/algaeproperty.h"
#include "SPH/isphparticles.h"


class Algae : public BaseSphParticle
{
public:
    Algae(std::shared_ptr<AlgaeProperty> _property);

    Algae(std::shared_ptr<AlgaeProperty> _property, Mesh _mesh);

    virtual ~Algae();

    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual AlgaeProperty *GetProperty();

    virtual void MapCudaGLResources();

    virtual void ReleaseCudaGLResources();




protected:
    void InitAlgaeAsMesh();
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();


    // specfic simulation Data
    std::shared_ptr<AlgaeProperty> m_property;
};

#endif // ALGAE_H
