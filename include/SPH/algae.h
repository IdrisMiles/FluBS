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



    float *GetPrevPressurePtr();
    void ReleasePrevPressurePtr();

    float *GetEnergyPtr();
    void ReleaseEnergyPtr();

    float *GetIlluminationPtr();
    void ReleaseIlluminationPtr();

    QOpenGLBuffer &GetIllumBO();

    void GetBioluminescentIntensities(std::vector<float> &_bio);
    void SetBioluminescentIntensities(const std::vector<float> &_bio);



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

    float *d_prevPressurePtr;
    float *d_energyPtr;
    float *d_illumPtr;

    bool m_illumMapped;
    QOpenGLBuffer m_illumBO;
    cudaGraphicsResource *m_illumBO_CUDA;
};

#endif // ALGAE_H
