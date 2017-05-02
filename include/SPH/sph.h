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

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::vector<std::shared_ptr<BaseSphParticle>> _sphParticles);

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<Fluid> _fluid);

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::vector<std::shared_ptr<Fluid>> _fluid);

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<Rigid> _rigid);

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::vector<std::shared_ptr<Rigid>> _rigid);

    void ResetProperties(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<Algae> _algae);


    template<typename T, typename ... Targs>
    void ResetProperties(const FluidSolverProperty &_solverProps,
                         T _sphParticles,
                         Targs ... args)
    {
        ResetProperties(_solverProps, _sphParticles);
        ResetProperties(_solverProps, args ...);
    }

    void ResetTotalForce(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    void InitFluidAsCube(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    void InitAlgaeIllumination(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<Algae> _algae);


    //--------------------------------------------------------------------------------------------------------------------
    // Compute Particle Hash functions

    void ComputeHash(const FluidSolverProperty &_solverProps,
                     std::shared_ptr<BaseSphParticle> _fluid);

    template<typename T>
    void ComputeHash(const FluidSolverProperty &_solverProps,
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
    void SortParticlesByHash(std::shared_ptr<Algae> _sphParticles);

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

    void ComputeParticleScatterIds(const FluidSolverProperty &_solverProps,
                                   std::shared_ptr<BaseSphParticle> _sphParticles);

    template<typename T>
    void ComputeParticleScatterIds(const FluidSolverProperty &_solverProps,
                                   std::vector<T> _sphParticles)
    {
        for(auto &&f : _sphParticles)
        {
            ComputeParticleScatterIds(_solverProps, f);
        }
    }

//    template<typename T, typename ... Targs>
//    void ComputeParticleScatterIds(const FluidSolverProperty &_solverProps,
//                                   T _sphParticles,
//                                   Targs ... args)
//    {
//        ComputeParticleScatterIds(_solverProps, _sphParticles);
//        ComputeParticleScatterIds(_solverProps, args ...);
//    }

    //--------------------------------------------------------------------------------------------------------------------
    // Compute max cell occupancy functions

    void ComputeMaxCellOccupancy(const FluidSolverProperty &_solverProps,
                                 std::shared_ptr<BaseSphParticle> _fluid,
                                 unsigned int &_maxCellOcc);

    template<typename T>
    void ComputeMaxCellOccupancy(const FluidSolverProperty &_solverProps,
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

    void ComputeParticleVolume(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<Rigid> _rigid);

    void ComputeParticleVolume(const FluidSolverProperty &_solverProps,
                               std::vector<std::shared_ptr<Rigid>> _rigid);

    //--------------------------------------------------------------------------------------------------------------------
    // Compute Density functions

    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        const bool accumulate = false);

    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<BaseSphParticle> _fluidContributer,
                        const bool accumulate = false);

    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<Rigid> _rigid,
                        const bool accumulate = false);

    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                        const bool accumulate = false);

    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<Rigid>> _rigids,
                        const bool accumulate = false);

    template<typename T, typename ... Targs>
    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        const bool accumulate,
                        T _fluidContributer,
                        Targs ... args)
    {
        ComputeDensity(_solverProps, _fluid, _fluidContributer, accumulate);
        ComputeDensity(_solverProps, _fluid, accumulate, args ...);
    }

    //--------------------------------------------------------------------------------------------------------------------

    void ComputePressure(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _particles);

    void ComputePressure(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<Fluid> _fluid);

    void ComputePressure(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<Algae> _algae,
                         std::shared_ptr<Fluid> _fluid);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _fluid
    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              const bool accumulate = false);

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _fluidContributer
    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<BaseSphParticle> _fluidContributer,
                              const bool accumulate = false);

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _rigid
    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<Rigid> _rigid,
                              const bool accumulate = false);

    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                              const bool accumulate = false);

    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<Rigid>> _rigids,
                              const bool accumulate = false);

    template<typename T, typename ... Targs>
    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              const bool accumulate,
                              T _fluidContributer,
                              Targs ... args)
    {
        ComputePressureForce(_solverProps, _fluid, _fluidContributer, accumulate);
        ComputePressureForce(_solverProps, _fluid, accumulate, args ...);
    }

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeViscForce(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Fluid> _fluid);

    void ComputeViscForce(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<Rigid> _rigid);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeSurfaceTensionForce(const FluidSolverProperty &_solverProps,
                                    std::shared_ptr<Fluid> _fluid);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeForces(const FluidSolverProperty &_solverProps,
                       std::shared_ptr<Fluid> _fluid,
                       const bool pressure = true,
                       const bool viscosity = true,
                       const bool surfTen = true,
                       const bool accumulate = false);

    //--------------------------------------------------------------------------------------------------------------------

    void ComputeTotalForce(const FluidSolverProperty &_solverProps,
                           std::shared_ptr<Fluid> _fluid,
                           const bool accumulatePressure = true,
                           const bool accumulateViscous = true,
                           const bool accumulateSurfTen = true,
                           const bool accumulateExternal = true,
                           const bool accumulateGravity = true);

    //--------------------------------------------------------------------------------------------------------------------

    void Integrate(const FluidSolverProperty &_solverProps,
                   std::shared_ptr<BaseSphParticle> _particles);

    //--------------------------------------------------------------------------------------------------------------------

    void HandleBoundaries(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<BaseSphParticle> _fluid);

    //--------------------------------------------------------------------------------------------------------------------
    // Algae movement stuff

    // Advection force = surrounding particles total forces,
    // use advectiion, pressure, etc... forces to move particle
    void ComputeAdvectionForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _particles,
                               std::shared_ptr<Fluid> _advector,
                               const bool accumulate = false);


    //--------------------------------------------------------------------------------------------------------------------
    // OR

    // move particle based on surrounding particles velocities
    void AdvectParticle(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _particles,
                        std::shared_ptr<Fluid> _advector);


    //--------------------------------------------------------------------------------------------------------------------

    // compute bioluminescence from stored pressures, or compute pressure aswell
    void ComputeBioluminescence(const FluidSolverProperty &_solverProps,
                                std::shared_ptr<Algae> _algae);

    //--------------------------------------------------------------------------------------------------------------------

    // PCI SPH
    namespace pci
    {
        void PredictIntegrate(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<Fluid> _fluid);

        void PredictDensity(const FluidSolverProperty &_solverProps,
                            std::shared_ptr<Fluid> _fluid);

        void predictDensityVariation(const FluidSolverProperty &_solverProps,
                                     std::shared_ptr<Fluid> _fluid);

        void ComputeMaxDensityVariation(const FluidSolverProperty &_solverProps,
                                        std::shared_ptr<Fluid> _fluid,
                                        float &_maxDenVar);

        void UpdatePressure(const FluidSolverProperty &_solverProps,
                            std::shared_ptr<Fluid> _fluid);

        void ComputePressureForce(const FluidSolverProperty &_solverProps,
                                  std::shared_ptr<Fluid> _fluid);


    }

}

#endif // SPH_H
