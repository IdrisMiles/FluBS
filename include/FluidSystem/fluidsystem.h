#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H


#include "FluidSystem/fluidsolverproperty.h"
#include "SPH/sph.h"
#include "SPH/fluid.h"
#include "SPH/rigid.h"
#include "SPH/fluidproperty.h"


/// @author Idris Miles
/// @version 1.0


class Poly6Kernel;
class SpikyKernel;

/// @class FluidSystem, this class implements a custom fluid solver
class FluidSystem
{
public:
    FluidSystem(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty = nullptr);
    FluidSystem(const FluidSystem &_FluidSystem);
    ~FluidSystem();

    void SetContainer(std::shared_ptr<Rigid> _container);
    void AddFluid(std::shared_ptr<Fluid> _fluid);
    void AddRigid(std::shared_ptr<Rigid> _rigid);
    void AddAlgae(std::shared_ptr<Algae> _algae);
    void AddFluidSolverProperty(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty);

    virtual void InitialiseSim();
    virtual void ResetSim();
    virtual void StepSim();



private:
    void ResetRigid(std::shared_ptr<Rigid> _rigid);
    void ResetFluid(std::shared_ptr<Fluid> _fluid);
    void ResetAlgae(std::shared_ptr<Algae> _algae);
    void GenerateDefaultContainer();

    std::shared_ptr<Algae> m_algae;
    std::shared_ptr<Fluid> m_fluid;
    std::shared_ptr<Rigid> m_container;
    std::vector<std::shared_ptr<Rigid>> m_staticRigids;
    std::vector<std::shared_ptr<Rigid>> m_activeRigids;
    std::shared_ptr<FluidSolverProperty> m_fluidSolverProperty;
    Poly6Kernel *m_poly6Kernel;
    SpikyKernel *m_spikyKernel;

    int m_frame;
    bool m_cache;

};

#endif // FLUIDSYSTEM_H
