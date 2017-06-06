#ifndef SPH_H
#define SPH_H

//--------------------------------------------------------------------------------------------------------------

#include "SPH/basesphparticle.h"
#include "SPH/fluid.h"
#include "SPH/rigid.h"
#include "SPH/algae.h"
#include "SPH/sphGPU.h"
#include "SPH/fluidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------

/// @namespace sph
/// @brief sph namespace hold library of sph functions.
/// Typical interface to sph functions void func(FluidSolverProperty, SphParticle)
namespace sph
{
    //--------------------------------------------------------------------------------------------------------------------
    // Reset Property functions

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::vector<std::shared_ptr<BaseSphParticle>> _sphParticles);

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::shared_ptr<Fluid> _fluid);

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::vector<std::shared_ptr<Fluid>> _fluid);

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::shared_ptr<Rigid> _rigid);

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::vector<std::shared_ptr<Rigid>> _rigid);

    /// @brief Method to reset sph properties before starting next simulation step
    void ResetProperties(const FluidSolverProperty &_solverProps, std::shared_ptr<Algae> _algae);


    /// @brief Method to reset sph properties before starting next simulation step,
    /// using variadic templates so we can perform operation on multiple sph particles with one function call
    template<typename T, typename ... Targs>
    void ResetProperties(const FluidSolverProperty &_solverProps, T _sphParticles, Targs ... args)
    {
        ResetProperties(_solverProps, _sphParticles);
        ResetProperties(_solverProps, args ...);
    }


    /// @brief Method to reset total force on particles
    void ResetTotalForce(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _sphParticles);
    //--------------------------------------------------------------------------------------------------------------------
    /// Initialisation methods

    /// @brief Method to initialse fluid in cube volume
    void InitFluidAsCube(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief Method to initialise algae particles illumination
    void InitAlgaeIllumination(const FluidSolverProperty &_solverProps, std::shared_ptr<Algae> _algae);

    /// @brief Method to initialise sph particle ids
    void InitSphParticleIds(std::shared_ptr<BaseSphParticle> _sphParticles);


    //--------------------------------------------------------------------------------------------------------------------
    // Compute Particle Hash functions

    /// @brief Method to compute sph particle hash ids giving solve properties
    void ComputeHash(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief Method to compute sph particle hash ids giving solve properties
    template<typename T>
    void ComputeHash(const FluidSolverProperty &_solverProps, std::vector<T> _fluid)
    {
        for(auto &&f : _fluid)
        {
            ComputeHash(_solverProps, f);
        }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Sort Particle functions

    /// @brief Method to sort sph particles by their hash ids
    void SortParticlesByHash(std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief Method to sort algae particles by their hash ids
    void SortParticlesByHash(std::shared_ptr<Algae> _sphParticles);

    /// @brief Method to sort vector of particles by their hash ids
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

    /// @brief Method to compute particle scatter addresses
    void ComputeParticleScatterIds(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief Method to compute vector particles scatter addresses
    template<typename T>
    void ComputeParticleScatterIds(const FluidSolverProperty &_solverProps, std::vector<T> _sphParticles)
    {
        for(auto &&f : _sphParticles)
        {
            ComputeParticleScatterIds(_solverProps, f);
        }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Compute max cell occupancy functions

    /// @brief Method to compute the max cell occupancy of particles in the solver
    void ComputeMaxCellOccupancy(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _fluid, unsigned int &_maxCellOcc);

    /// @brief Method to compute the max cell occupancy of a vector of particles in the solver
    template<typename T>
    void ComputeMaxCellOccupancy(const FluidSolverProperty &_solverProps, std::vector<T> _fluid, unsigned int &_maxCellOcc)
    {
        for(auto &&f : _fluid)
        {
            ComputeMaxCellOccupancy(_solverProps, f, _maxCellOcc);
        }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Compute particle volume functions - for Rigid

    /// @brief Method to compute Rigid particle volumes
    void ComputeParticleVolume(const FluidSolverProperty &_solverProps, std::shared_ptr<Rigid> _rigid);

    /// @brief Method to compute a vector of Rigid particle volumes
    void ComputeParticleVolume(const FluidSolverProperty &_solverProps, std::vector<std::shared_ptr<Rigid>> _rigid);

    //--------------------------------------------------------------------------------------------------------------------
    // Compute Density functions

    /// @brief Method to calculate density of sph particles
    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        const bool accumulate = false);

    /// @brief Method to calculate density of sph particles contributed by neighbouring sph particles sets
    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<BaseSphParticle> _fluidContributer,
                        const bool accumulate = false);

    /// @brief Method to calculate density of sph particles contributed by neighbouring rigid particles
    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<Rigid> _rigid,
                        const bool accumulate = false);

    /// @brief Method to calculate density of sph particles contributed by neighbouring vector of sph particles
    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                        const bool accumulate = false);

    /// @brief Method to calculate density of sph particles contributed by neighbouring vector of rigid particles
    void ComputeDensity(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<Rigid>> _rigids,
                        const bool accumulate = false);

    /// @brief Method to calculate density of sph particles contributed by arbitrary number neighbouring particles
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

    /// @brief Method to compute pressure of sph particles
    void ComputePressure(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _particles);

    /// @brief Method to compute pressure of fluid particles
    void ComputePressure(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<Fluid> _fluid);

    /// @brief Method to compute pressure of algae particles contributed by neighbouring fluid
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

    /// @brief Method to compute preesure force on sph particles contributed by vector of neighbour sph particles
    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                              const bool accumulate = false);

    /// @brief Method to compute preesure force on sph particles contributed by vector of neighbour rigid particles
    void ComputePressureForce(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<Rigid>> _rigids,
                              const bool accumulate = false);

    /// @brief Method to compute preesure force on sph particles contributed by arbitrary number/types of neighbour sph particles
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

    /// @brief Method to compute viscosity force on fluid particles
    void ComputeViscForce(const FluidSolverProperty &_solverProps, std::shared_ptr<Fluid> _fluid);

    /// @brief Method to compute viscosity force on fluid contributed by neighbouring rigid particles.
    void ComputeViscForce(const FluidSolverProperty &_solverProps, std::shared_ptr<Fluid> _fluid, std::shared_ptr<Rigid> _rigid);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to compute surface tnesion force on fluid particles
    void ComputeSurfaceTensionForce(const FluidSolverProperty &_solverProps, std::shared_ptr<Fluid> _fluid);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to compute various forces acting on fluid particles
    void ComputeForces(const FluidSolverProperty &_solverProps,
                       std::shared_ptr<Fluid> _fluid,
                       const bool pressure = true,
                       const bool viscosity = true,
                       const bool surfTen = true,
                       const bool accumulate = false);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to compute total force acting on fluid particles
    void ComputeTotalForce(const FluidSolverProperty &_solverProps,
                           std::shared_ptr<Fluid> _fluid,
                           const bool accumulatePressure = true,
                           const bool accumulateViscous = true,
                           const bool accumulateSurfTen = true,
                           const bool accumulateExternal = true,
                           const bool accumulateGravity = true);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to integrate fluid particles
    void Integrate(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _particles);

    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to explicitly handle boundaries
    void HandleBoundaries(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _fluid);

    //--------------------------------------------------------------------------------------------------------------------
    // Algae movement stuff

    /// @brief Method to compute advection force on sph particles from neighbouring fluid particles.
    void ComputeAdvectionForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _particles,
                               std::shared_ptr<Fluid> _advector,
                               const bool accumulate = false);


    /// @brief Method to directly advect sph particles using neighbouring fluid particles
    void AdvectParticle(const FluidSolverProperty &_solverProps, std::shared_ptr<BaseSphParticle> _particles, std::shared_ptr<Fluid> _advector);


    //--------------------------------------------------------------------------------------------------------------------

    /// @brief Method to compute biolumenscence on algae particles
    void ComputeBioluminescence(const FluidSolverProperty &_solverProps, std::shared_ptr<Algae> _algae);

    //--------------------------------------------------------------------------------------------------------------------

}

//--------------------------------------------------------------------------------------------------------------

#endif // SPH_H
