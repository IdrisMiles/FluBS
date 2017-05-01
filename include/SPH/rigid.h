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

    void UpdateMesh(Mesh &_mesh);

    virtual void SetupSolveSpecs(const FluidSolverProperty &_solverProps);

    virtual RigidProperty* GetProperty();

    virtual void MapCudaGLResources();
    virtual void ReleaseCudaGLResources();

    float *GetVolumePtr();
    void ReleaseVolumePtr();




    virtual void GetPositions(std::vector<glm::vec3> &_pos);
    virtual void GetVelocities(std::vector<glm::vec3> &_vel);
    virtual void GetParticleIds(std::vector<int> &_ids);

    virtual void SetPositions(const std::vector<glm::vec3> &_pos);
    virtual void SetVelocities(const std::vector<glm::vec3> &_vel);
    virtual void SetParticleIds(const std::vector<int> &_ids);



protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();


    // specfic simulation Data
    std::shared_ptr<RigidProperty> m_property;
    float* d_volumePtr;

};

#endif // RIGID_H
