#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H

#include "include/fluid.h"
#include "include/sphsolverGPU.h"
#include "include/fluidproperty.h"
#include "include/fluidsolverproperty.h"


class FluidSystem
{
public:
    FluidSystem(std::shared_ptr<SPHSolverGPU> _fluidSolver = nullptr,
                std::shared_ptr<Fluid> _fluid = nullptr,
                std::shared_ptr<FluidSolverProperty> _fluidSolverProperty = nullptr,
                std::shared_ptr<FluidProperty> _fluidProperty = nullptr);
    FluidSystem(const FluidSystem &_FluidSystem);
    ~FluidSystem();

    void AddFluidSolver(std::shared_ptr<SPHSolverGPU> _fluidSolver);
    void AddFluid(std::shared_ptr<Fluid> _fluid);
    void AddFluidSolverProperty(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty);
    void AddFluidProperty(std::shared_ptr<FluidProperty> _fluidProperty);

    void InitialiseSim();
    void ResetSim();
    void StepSimulation();

private:

    std::shared_ptr<Fluid> m_fluid;
    std::shared_ptr<SPHSolverGPU> m_fluidSolver;
    std::shared_ptr<FluidProperty> m_fluidProperty;
    std::shared_ptr<FluidSolverProperty> m_fluidSolverProperty;

};

#endif // FLUIDSYSTEM_H
