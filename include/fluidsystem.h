#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H

#include "include/fluid.h"
#include "include/sphsolverGPU.h"
#include "include/fluidproperty.h"
#include "include/fluidsolverproperty.h"
#include "include/sph.h"


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

};

#endif // FLUIDSYSTEM_H
