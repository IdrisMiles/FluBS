#ifndef SPH_H
#define SPH_H

#include "Fluid/isphparticles.h"
#include "Fluid/fluid.h"
//#include "Fluid/"
#include "Rigid/rigid.h"
#include "SPH/sphGPU.h"
#include "Fluid/fluidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"

namespace sph
{
    void ResetProperties(std::shared_ptr<ISphParticles> _sphParticles,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::shared_ptr<Rigid> _rigid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void InitFluidAsCube(std::shared_ptr<ISphParticles> _sphParticles,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    //--------------------------------------------------------------------------------------
    void ComputeHash(std::shared_ptr<ISphParticles> _fluid,
                     std::shared_ptr<FluidSolverProperty> _solverProps);

    void SortParticlesByHash(std::shared_ptr<ISphParticles> _sphParticles);

    void ComputeParticleScatterIds(std::shared_ptr<ISphParticles> _sphParticles,
                                   std::shared_ptr<FluidSolverProperty> _solverProps);

    void ComputeMaxCellOccupancy(std::shared_ptr<ISphParticles> _fluid,
                                 std::shared_ptr<FluidSolverProperty> _solverProps,
                                 unsigned int &_maxCellOcc);

    //--------------------------------------------------------------------------------------

    void ComputeParticleVolume(std::shared_ptr<Rigid> _rigid,
                               std::shared_ptr<FluidSolverProperty> _solverProps);

    //--------------------------------------------------------------------------------------

    void ComputeDensityFluid(std::shared_ptr<ISphParticles> _fluid,
                             std::shared_ptr<FluidSolverProperty> _solverProps,
                             const bool accumulate = false);

    void ComputeDensityFluidFluid(std::shared_ptr<ISphParticles> _fluid,
                                  std::shared_ptr<ISphParticles> _fluidContributer,
                                  std::shared_ptr<FluidSolverProperty> _solverProps,
                                  const bool accumulate = false);

    void ComputeDensityFluidRigid(std::shared_ptr<ISphParticles> _fluid,
                                  std::shared_ptr<Rigid> _rigid,
                                  std::shared_ptr<FluidSolverProperty> _solverProps,
                                  const bool accumulate = false);


    void ComputePressureFluid(std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

//--------------------------------------------------------------------------------------

    void ComputePressureForceFluid(std::shared_ptr<ISphParticles> _fluid,
                                   std::shared_ptr<FluidSolverProperty> _solverProps,
                                   const bool accumulate = false);

    void ComputePressureForceFluidFluid(std::shared_ptr<ISphParticles> _fluid,
                                        std::shared_ptr<ISphParticles> _fluidContributer,
                                        std::shared_ptr<FluidSolverProperty> _solverProps,
                                        const bool accumulate = false);

    void ComputePressureForceFluidRigid(std::shared_ptr<ISphParticles> _fluid,
                                        std::shared_ptr<Rigid> _rigid,
                                        std::shared_ptr<FluidSolverProperty> _solverProps,
                                        const bool accumulate = false);


    //--------------------------------------------------------------------------------------

    void ComputeViscForce(std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<FluidSolverProperty> _solverProps);

    void ComputeSurfaceTensionForce(std::shared_ptr<Fluid> _fluid,
                                    std::shared_ptr<FluidSolverProperty> _solverProps);

    void ComputeTotalForce(std::shared_ptr<Fluid> _fluid,
                           std::shared_ptr<FluidSolverProperty> _solverProps);

    //--------------------------------------------------------------------------------------

    void Integrate(std::shared_ptr<Fluid> _fluid,
                   std::shared_ptr<FluidSolverProperty> _solverProps);

    void HandleBoundaries(std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<FluidSolverProperty> _solverProps);
}

#endif // SPH_H
