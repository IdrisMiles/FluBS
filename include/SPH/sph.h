#ifndef SPH_H
#define SPH_H

#include "SPH/isphparticles.h"
#include "SPH/fluid.h"
#include "SPH/rigid.h"
#include "SPH/algae.h"
#include "SPH/sphGPU.h"
#include "SPH/fluidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"

namespace sph
{
    //--------------------------------------------------------------------------------------------------------------------
    // Reset Property functions

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::vector<std::shared_ptr<BaseSphParticle>> _sphParticles);

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<Fluid> _fluid);

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::vector<std::shared_ptr<Fluid>> _fluid);

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<Rigid> _rigid);

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::vector<std::shared_ptr<Rigid>> _rigid);

    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<Algae> _algae);


    template<typename T, typename ... Targs>
    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
                         T _sphParticles,
                         Targs ... args)
    {
        ResetProperties(_solverProps, _sphParticles);
        ResetProperties(_solverProps, args ...);
    }

    void ResetTotalForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    void InitFluidAsCube(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);


    //--------------------------------------------------------------------------------------------------------------------
    // Compute Particle Hash functions

    void ComputeHash(std::shared_ptr<FluidSolverProperty> _solverProps,
                     std::shared_ptr<BaseSphParticle> _fluid);

    template<typename T>
    void ComputeHash(std::shared_ptr<FluidSolverProperty> _solverProps,
                     std::vector<T> _fluid)
    {
        for(auto &&f : _fluid)
        {
            ComputeHash(_solverProps, f);
        }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Sort Particle functions

    void SortParticlesByHash(std::shared_ptr<BaseSphParticle> _sphParticles);

    template<typename T>
    void SortParticlesByHash(std::vector<T> _sphParticles)
    {
        for(auto &&f : _sphParticles)
        {
            SortParticlesByHash(f);
        }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Compute scatter address functions

    void ComputeParticleScatterIds(std::shared_ptr<FluidSolverProperty> _solverProps,
                                   std::shared_ptr<BaseSphParticle> _sphParticles);

    template<typename T>
    void ComputeParticleScatterIds(std::shared_ptr<FluidSolverProperty> _solverProps,
                                   std::vector<T> _sphParticles)
    {
        for(auto &&f : _sphParticles)
        {
            ComputeParticleScatterIds(_solverProps, f);
        }
    }

//    template<typename T, typename ... Targs>
//    void ComputeParticleScatterIds(std::shared_ptr<FluidSolverProperty> _solverProps,
//                                   T _sphParticles,
//                                   Targs ... args)
//    {
//        ComputeParticleScatterIds(_solverProps, _sphParticles);
//        ComputeParticleScatterIds(_solverProps, args ...);
//    }

    //--------------------------------------------------------------------------------------------------------------------
    // Compute max cell occupancy functions

    void ComputeMaxCellOccupancy(std::shared_ptr<FluidSolverProperty> _solverProps,
                                 std::shared_ptr<BaseSphParticle> _fluid,
                                 unsigned int &_maxCellOcc);

    template<typename T>
    void ComputeMaxCellOccupancy(std::shared_ptr<FluidSolverProperty> _solverProps,
                                 std::vector<T> _fluid,
                                 unsigned int &_maxCellOcc)
    {
        for(auto &&f : _fluid)
        {
            ComputeMaxCellOccupancy(_solverProps, f, _maxCellOcc);
        }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Compute particle volume functions - for Rigid

    void ComputeParticleVolume(std::shared_ptr<FluidSolverProperty> _solverProps,
                               std::shared_ptr<Rigid> _rigid);

    void ComputeParticleVolume(std::shared_ptr<FluidSolverProperty> _solverProps,
                               std::vector<std::shared_ptr<Rigid>> _rigid);

    //--------------------------------------------------------------------------------------------------------------------
    // Compute Density functions

    void ComputeDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<BaseSphParticle> _fluidContributer,
                        const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<Rigid> _rigid,
                        const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                        const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<Rigid>> _rigids,
                        const bool accumulate = false);

    template<typename T, typename ... Targs>
    void ComputeDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        const bool accumulate,
                        T _fluidContributer,
                        Targs ... args)
    {
        ComputeDensity(_solverProps, _fluid, _fluidContributer, accumulate);
        ComputeDensity(_solverProps, _fluid, accumulate, args ...);
    }

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressure(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<BaseSphParticle> _particles);

    void ComputePressure(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<Fluid> _fluid);

    void ComputePressure(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<Algae> _algae);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _fluid
    void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              const bool accumulate = false);

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _fluidContributer
    void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<BaseSphParticle> _fluidContributer,
                              const bool accumulate = false);

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _rigid
    void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<Rigid> _rigid,
                              const bool accumulate = false);

    void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                              const bool accumulate = false);

    void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<Rigid>> _rigids,
                              const bool accumulate = false);

    template<typename T, typename ... Targs>
    void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              const bool accumulate,
                              T _fluidContributer,
                              Targs ... args)
    {
        ComputePressureForce(_solverProps, _fluid, _fluidContributer, accumulate);
        ComputePressureForce(_solverProps, _fluid, accumulate, args ...);
    }

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeViscForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                          std::shared_ptr<Fluid> _fluid);

    void ComputeViscForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                          std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<Rigid> _rigid);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeSurfaceTensionForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                                    std::shared_ptr<Fluid> _fluid);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeForces(std::shared_ptr<FluidSolverProperty> _solverProps,
                       std::shared_ptr<Fluid> _fluid,
                       const bool pressure = true,
                       const bool viscosity = true,
                       const bool surfTen = true,
                       const bool accumulate = false);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeTotalForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                           std::shared_ptr<Fluid> _fluid,
                           const bool accumulatePressure = true,
                           const bool accumulateViscous = true,
                           const bool accumulateSurfTen = true,
                           const bool accumulateExternal = true,
                           const bool accumulateGravity = true);

    //--------------------------------------------------------------------------------------------------------------------

    void Integrate(std::shared_ptr<FluidSolverProperty> _solverProps,
                   std::shared_ptr<Fluid> _fluid);

    //--------------------------------------------------------------------------------------------------------------------

    void HandleBoundaries(std::shared_ptr<FluidSolverProperty> _solverProps,
                          std::shared_ptr<BaseSphParticle> _fluid);

    //--------------------------------------------------------------------------------------------------------------------
    // Algae movement stuff

    // Advection force = surrounding particles total forces,
    // use advectiion, pressure, etc... forces to move particle
    void ComputeAdvectionForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                               std::shared_ptr<BaseSphParticle> _particles,
                               std::shared_ptr<Fluid> _advector);


    //--------------------------------------------------------------------------------------------------------------------
    // OR

    // move particle based on surrounding particles velocities
    void AdvectParticle(std::shared_ptr<FluidSolverProperty> _solverProps,
                        std::shared_ptr<BaseSphParticle> _particles,
                        std::shared_ptr<Fluid> _advector);


    //--------------------------------------------------------------------------------------------------------------------

    // compute bioluminescence from stored pressures, or compute pressure aswell
    void ComputeBioluminescence(std::shared_ptr<FluidSolverProperty> _solverProps,
                                std::shared_ptr<Algae> _algae,
                                bool accumulate);

    //--------------------------------------------------------------------------------------------------------------------

    // PCI SPH
    namespace pci
    {
        void PredictIntegrate(std::shared_ptr<FluidSolverProperty> _solverProps,
                              std::shared_ptr<Fluid> _fluid);

        void PredictDensity(std::shared_ptr<FluidSolverProperty> _solverProps,
                            std::shared_ptr<Fluid> _fluid);

        void predictDensityVariation(std::shared_ptr<FluidSolverProperty> _solverProps,
                                     std::shared_ptr<Fluid> _fluid);

        void ComputeMaxDensityVariation(std::shared_ptr<FluidSolverProperty> _solverProps,
                                        std::shared_ptr<Fluid> _fluid,
                                        float &_maxDenVar);

        void UpdatePressure(std::shared_ptr<FluidSolverProperty> _solverProps,
                            std::shared_ptr<Fluid> _fluid);

        void ComputePressureForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                                  std::shared_ptr<Fluid> _fluid);


    }

}

#endif // SPH_H
