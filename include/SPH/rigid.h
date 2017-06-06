#ifndef RIGID_H
#define RIGID_H

//--------------------------------------------------------------------------------------------------------------


// sph includes
#include "SPH/basesphparticle.h"
#include "SPH/rigidproperty.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------



/// @class Rigid
/// @brief Rigid inherits from BaseSphParticle, this class holds all data and functionality specific to rigid body particles
class Rigid : public BaseSphParticle
{
public:
    /// @brief Constructor
    Rigid(std::shared_ptr<RigidProperty> _rigidProperty, Mesh _mesh, std::string _name = "rigid", std::string _type = "rigid");

    /// @brief Deconstructor
    virtual ~Rigid();

    //------------------------------------------------------------------------------------------------------------

    /// @brief Method to get RigidProperty associated with this instance
    virtual RigidProperty* GetProperty();

    /// @brief Method to set an instances properties
    void SetProperty(RigidProperty _property);

    /// @brief Method to set the type of rigid, i.e. cube, sphere, mesh
    void SetType(std::string type);

    /// @brief Method to get the type of this rigid instance
    std::string GetType();

    /// @brief Method to set the file name this rigid was loaded from if loading a mesh
    void SetFileName(std::string file);

    /// @brief Method to get the file name of the mesh used to load this rigid
    std::string GetFileName();

    /// @brief Method to set up certain internal data dependant on the solver properties, such as the hash id, cell occupancy arrays
    virtual void SetupSolveSpecs(const FluidSolverProperty &_solverProps);

    /// @brief Method update the mesh particles
    void UpdateMesh(Mesh &_mesh, const glm::vec3 &_pos = glm::vec3(0.0f, 0.0f, 0.0f), const glm::vec3 &_rot = glm::vec3(0.0f, 0.0f, 0.0f));
    /// @brief Method update the mesh particles
    void UpdateMesh(const glm::vec3 &_pos = glm::vec3(0.0f, 0.0f, 0.0f), const glm::vec3 &_rot = glm::vec3(0.0f, 0.0f, 0.0f));
    /// @brief Method to get the position of this rigid
    glm::vec3 GetPos();
    /// @brief Method to get the rotation of this rigid
    glm::vec3 GetRot();

    /// @brief Method to get GPU particle data that can be used directly in CUDA kernel,
    /// this is used within the sph library
    RigidGpuData GetRigidGpuData();


    /// @brief Method to map CUDA resources in one call, so we can use memory also being used by OpenGL
    virtual void MapCudaGLResources();

    /// @brief Method to Release CUDA OpenGL resources all in one call.
    virtual void ReleaseCudaGLResources();

    /// @brief Method to get pointer to CUDA memory holding volume data
    float *GetVolumePtr();



protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();

    virtual void UpdateCUDAMemory();


    std::string m_type;
    std::string m_fileName;
    glm::vec3 m_pos;
    glm::vec3 m_rot;


    // specfic simulation Data
    std::shared_ptr<RigidProperty> m_property;
    float* d_volumePtr;

};

//--------------------------------------------------------------------------------------------------------------

#endif // RIGID_H
