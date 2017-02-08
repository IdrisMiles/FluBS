#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H

#include "Fluid/fluid.h"
#include "Fluid/fluidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"
#include "SPH/sph.h"

class Poly6Kernel;
class SpikyKernel;

class FluidSystem
{
public:
    FluidSystem(std::shared_ptr<Fluid> _fluid = nullptr,
                std::shared_ptr<FluidSolverProperty> _fluidSolverProperty = nullptr);
    FluidSystem(const FluidSystem &_FluidSystem);
    ~FluidSystem();

    void AddFluid(std::shared_ptr<Fluid> _fluid);
    void AddAlgae(std::shared_ptr<Fluid> _algae);
    void AddFluidSolverProperty(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty);

    void InitialiseSim();
    void ResetSim();
    void StepSimulation();



private:
    std::shared_ptr<Fluid> m_algae;
    std::shared_ptr<Fluid> m_fluid;
    std::shared_ptr<FluidSolverProperty> m_fluidSolverProperty;
    Poly6Kernel *m_poly6Kernel;
    SpikyKernel *m_spikyKernel;

};

#endif // FLUIDSYSTEM_H
