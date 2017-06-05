#ifndef ALGAE_H
#define ALGAE_H

//--------------------------------------------------------------------------------------------------------------


#include "SPH/algaeproperty.h"
#include "SPH/basesphparticle.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @data 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class Algae
/// @brief Algae inherits from BaseSphParticle, this class holds all data and functionality specific to algae particles
class Algae : public BaseSphParticle
{
public:
    /// @brief Constructor
    Algae(std::shared_ptr<AlgaeProperty> _property, std::string _name = "algae");

    /// @brief Constructor
    Algae(std::shared_ptr<AlgaeProperty> _property, Mesh _mesh, std::string _name = "algae");

    /// @brief Destructor
    virtual ~Algae();

    //------------------------------------------------------------------------------------------------------------

    /// @brief Method to set up certain internal data dependant on the solver properties, such as the hash id, cell occupancy arrays
    virtual void SetupSolveSpecs(const FluidSolverProperty &_solverProps);

    /// @brief Method to get AlgaeProperty associated with this instance
    virtual AlgaeProperty *GetProperty();

    /// @brief Method to set an instances properties
    void SetProperty(AlgaeProperty _property);

    /// @brief Method to get GPU particle data that can be used directly in CUDA kernel,
    /// this is used within the sph library
    AlgaeGpuData GetAlgaeGpuData();

    /// @brief Method to map CUDA resources in one call, so we can use memory also being used by OpenGL
    virtual void MapCudaGLResources();

    /// @brief Method to Release CUDA OpenGL resources all in one call.
    virtual void ReleaseCudaGLResources();


    /// @brief Method to get pointer to CUDA memory holding previous pressure data
    float *GetPrevPressurePtr();

    /// @brief Method to get pointer to CUDA memory holding illumination data
    float *GetIlluminationPtr();

    /// @brief Method to release CUDA resource to illumination data and give control back to OpenGL.
    void ReleaseIlluminationPtr();

    /// @brief Methof to get the illumination OpenGL Buffer.
    QOpenGLBuffer &GetIllumBO();


    /// @brief Method to download illumination data to CPU
    void GetBioluminescentIntensities(std::vector<float> &_bio);

    /// @brief Method to set the illumination data on the GPU from CPU data
    void SetBioluminescentIntensities(const std::vector<float> &_bio);


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

//--------------------------------------------------------------------------------------------------------------


#endif // ALGAE_H
