#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H


#include "FluidSystem/fluidsolverproperty.h"
#include "SPH/sph.h"
#include "SPH/fluid.h"
#include "SPH/rigid.h"
#include "SPH/fluidproperty.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class FluidSystem
/// @brief this class implements a custom fluid solver
class FluidSystem
{
public:
    /// @brief constructor
    FluidSystem(FluidSolverProperty _fluidSolverProperty = FluidSolverProperty());

    /// @brief constructor
    FluidSystem(const FluidSystem &_FluidSystem);

    /// @brief destrcutor
    ~FluidSystem();


    /// @brief Method to set solver properties
    void SetFluidSolverProperty(FluidSolverProperty _fluidSolverProperty);

    /// @brief Method to set custom container
    void SetContainer(std::shared_ptr<Rigid> _container);

    /// @brief Method to add fluid objects into solver
    void AddFluid(std::shared_ptr<Fluid> _fluid);

    /// @brief Method to add rigid objects into solver
    void AddRigid(std::shared_ptr<Rigid> _rigid);

    /// @brief Method to add algae objects into solver
    void AddAlgae(std::shared_ptr<Algae> _algae);


    /// @brief Method to remove fluid objects from solver
    void RemoveFluid(std::shared_ptr<Fluid> _fluid);

    /// @brief Method to remove rigid objects from solver
    void RemoveRigid(std::shared_ptr<Rigid> _rigid);

    /// @brief Method to remove algae objects from solver
    void RemoveAlgae(std::shared_ptr<Algae> _algae);

    /// @brief Method to get solver properties
    FluidSolverProperty GetProperty();

    /// @brief Method to get fluid object in solver
    std::shared_ptr<Fluid> GetFluid();

    /// @brief Method to get algae object in solver
    std::shared_ptr<Algae> GetAlgae();

    /// @brief Method to get vector of active rigid objects in solver
    std::vector<std::shared_ptr<Rigid>> GetActiveRigids();

    /// @brief Method to get vector of static rigid objects in solver
    std::vector<std::shared_ptr<Rigid>> GetStaticRigids();

    //----------------------------------------------------------------------------------------
    // These are methods that would be overloaded in custom solvers to achieve new features

    /// @brief Method to initialise sim
    virtual void InitialiseSim();

    /// @brief Method to reset sim
    virtual void ResetSim();

    /// @brief Method to step sim forward one frame
    virtual void StepSim();



private:
    void InitRigid(std::shared_ptr<Rigid> _rigid);
    void InitFluid(std::shared_ptr<Fluid> _fluid);
    void InitAlgae(std::shared_ptr<Algae> _algae);

    void ResetRigid(std::shared_ptr<Rigid> _rigid);
    void ResetFluid(std::shared_ptr<Fluid> _fluid);
    void ResetAlgae(std::shared_ptr<Algae> _algae);

    void GenerateDefaultContainer();

    std::shared_ptr<Algae> m_algae;
    std::shared_ptr<Fluid> m_fluid;
    std::shared_ptr<Rigid> m_container;
    std::vector<std::shared_ptr<Rigid>> m_staticRigids;
    std::vector<std::shared_ptr<Rigid>> m_activeRigids;
    FluidSolverProperty m_fluidSolverProperty;

    int m_frame;
    bool m_cache;

};

#endif // FLUIDSYSTEM_H
