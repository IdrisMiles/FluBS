#ifndef FLUID_H
#define FLUID_H

//--------------------------------------------------------------------------------------------------------------

// sph includes
#include "SPH/fluidproperty.h"
#include "SPH/basesphparticle.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class Fluid
/// @brief Fluid inherits from BaseSphParticle, this class holds all data and functionality specific to fluid particles
class Fluid : public BaseSphParticle
{

public:
    /// @brief Constructor
    Fluid(std::shared_ptr<FluidProperty> _fluidProperty, std::string _name = "fluid");

    /// @brief Constructor
    Fluid(std::shared_ptr<FluidProperty> _rigidProperty, Mesh _mesh, std::string _name = "fluid");

    /// @brief Destructor
    virtual ~Fluid();

    //------------------------------------------------------------------------------------------------------------

    /// @brief Method to set up certain internal data dependant on the solver properties, such as the hash id, cell occupancy arrays
    virtual void SetupSolveSpecs(const FluidSolverProperty &_solverProps);

    /// @brief Method to get FluidProperty associated with this instance
    virtual FluidProperty *GetProperty();

    /// @brief Method to set an instances properties
    virtual void SetProperty(FluidProperty _property);

    /// @brief Method to get GPU particle data that can be used directly in CUDA kernel,
    /// this is used within the sph library
    FluidGpuData GetFluidGpuData();

    /// @brief Method to map CUDA resources in one call, so we can use memory also being used by OpenGL
    void MapCudaGLResources();

    /// @brief Method to Release CUDA OpenGL resources all in one call.
    void ReleaseCudaGLResources();

    /// @brief Method to get pointer to CUDA memory holding viscous force data
    float3 *GetViscForcePtr();

    /// @brief Method to get pointer to CUDA memory holding Surface Tension Force data
    float3 *GetSurfTenForcePtr();

    //------------------------------------------------------------------------------------------------------------
    // PCI SPH stuff

    /// @brief Method to get pointer to CUDA memory holding predicted position data - for PCI SPH
    float3 *GetPredictPosPtr();

    /// @brief Method to get pointer to CUDA memory holding predicted velocity data - for PCI SPH
    float3 *GetPredictVelPtr();

    /// @brief Method to get pointer to CUDA memory holding density error data - for PCI SPH
    float *GetDensityErrPtr();


protected:
    void InitFluidAsMesh();
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUpGL();
    virtual void CleanUpCUDAMemory();

    virtual void UpdateCUDAMemory();


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

//--------------------------------------------------------------------------------------------------------------

#endif // FLUID_H
