#ifndef ALGAE_H
#define ALGAE_H

// sph includes
#include "SPH/algaeproperty.h"
#include "SPH/isphparticles.h"



class Algae : public BaseSphParticle
{
public:
    Algae(std::shared_ptr<AlgaeProperty> _property, std::string _name = "algae");

    Algae(std::shared_ptr<AlgaeProperty> _property, Mesh _mesh, std::string _name = "algae");

    virtual ~Algae();

    virtual void SetupSolveSpecs(const FluidSolverProperty &_solverProps);

    virtual AlgaeProperty *GetProperty();

    void SetProperty(AlgaeProperty _property);

    AlgaeGpuData GetAlgaeGpuData();

    virtual void MapCudaGLResources();

    virtual void ReleaseCudaGLResources();



    float *GetPrevPressurePtr();
    void ReleasePrevPressurePtr();

    float *GetIlluminationPtr();
    void ReleaseIlluminationPtr();

    QOpenGLBuffer &GetIllumBO();

    void GetBioluminescentIntensities(std::vector<float> &_bio);
    void SetBioluminescentIntensities(const std::vector<float> &_bio);


    virtual void GetPositions(std::vector<glm::vec3> &_pos);
    virtual void GetVelocities(std::vector<glm::vec3> &_vel);
    virtual void GetParticleIds(std::vector<int> &_ids);

    virtual void SetPositions(const std::vector<glm::vec3> &_pos);
    virtual void SetVelocities(const std::vector<glm::vec3> &_vel);
    virtual void SetParticleIds(const std::vector<int> &_ids);


protected:
    void InitAlgaeAsMesh();
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();

    virtual void UpdateCUDAMemory();


    // specfic simulation Data
    std::shared_ptr<AlgaeProperty> m_property;

    float *d_prevPressurePtr;
    float *d_illumPtr;

    bool m_illumMapped;
    QOpenGLBuffer m_illumBO;
    cudaGraphicsResource *m_illumBO_CUDA;
};

#endif // ALGAE_H
