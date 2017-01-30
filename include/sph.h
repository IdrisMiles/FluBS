#ifndef SPH_H
#define SPH_H

#include "include/fluid.h"
#include "include/boundary.h"
#include "include/sphGPU.h"
#include "fluidproperty.h"
#include "fluidsolverproperty.h"

namespace sph
{
    void ResetProperties(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void InitFluidAsCube(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void ComputeHash(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void SortParticlesByHash(std::shared_ptr<Fluid> _fluid);
    void ComputeParticleScatterIds(std::shared_ptr<Fluid> _fluid);
    void ComputeMaxCellOccupancy(std::shared_ptr<Fluid> _fluid, unsigned int &_maxCellOcc);
    void ComputePressure(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void ComputePressureForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void ComputeViscForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void ComputeSurfaceTensionForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void ComputeTotalForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void Integrate(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
    void HandleBoundaries(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps);
}

#endif // SPH_H
