#ifndef SPH_H
#define SPH_H

#include "SPH/isphparticles.h"
#include "SPH/fluid.h"
#include "SPH/rigid.h"
#include "SPH/sphGPU.h"
#include "SPH/fluidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"

namespace sph
{
    void ResetProperties(std::shared_ptr<BaseSphParticle> _sphParticles,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::vector<std::shared_ptr<BaseSphParticle>> _sphParticles,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::vector<std::shared_ptr<Fluid>> _fluid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::shared_ptr<Rigid> _rigid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void ResetProperties(std::vector<std::shared_ptr<Rigid>> _rigid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);


//    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps){;}
//    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
//                         std::shared_ptr<BaseSphParticle> _sphParticles){;}

//    template<typename T, typename ... Targs>
//    void ResetProperties(std::shared_ptr<FluidSolverProperty> _solverProps,
//                         T _sphParticles,
//                         Targs ... args)
//    {
//        ResetProperties(_sphParticles, _solverProps);
//        ResetProperties(_solverProps, args ...);
//    }

    void ResetTotalForce(std::shared_ptr<BaseSphParticle> _sphParticles,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    void InitFluidAsCube(std::shared_ptr<BaseSphParticle> _sphParticles,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

    //--------------------------------------------------------------------------------------
    void ComputeHash(std::shared_ptr<BaseSphParticle> _fluid,
                     std::shared_ptr<FluidSolverProperty> _solverProps);

    template<typename T>
    void ComputeHash(std::vector<T> _fluid,
                     std::shared_ptr<FluidSolverProperty> _solverProps)
    {
        for(auto &&f : _fluid)
        {
            ComputeHash(f, _solverProps);
        }
    }

    void SortParticlesByHash(std::shared_ptr<BaseSphParticle> _sphParticles);

    template<typename T>
    void SortParticlesByHash(std::vector<T> _sphParticles)
    {
        for(auto &&f : _sphParticles)
        {
            SortParticlesByHash(f);
        }
    }

    void ComputeParticleScatterIds(std::shared_ptr<BaseSphParticle> _sphParticles,
                                   std::shared_ptr<FluidSolverProperty> _solverProps);

    template<typename T>
    void ComputeParticleScatterIds(std::vector<T> _sphParticles,
                                   std::shared_ptr<FluidSolverProperty> _solverProps)
    {
        for(auto &&f : _sphParticles)
        {
            ComputeParticleScatterIds(f, _solverProps);
        }
    }

    void ComputeMaxCellOccupancy(std::shared_ptr<BaseSphParticle> _fluid,
                                 std::shared_ptr<FluidSolverProperty> _solverProps,
                                 unsigned int &_maxCellOcc);

    template<typename T>
    void ComputeMaxCellOccupancy(std::vector<T> _fluid,
                                 std::shared_ptr<FluidSolverProperty> _solverProps,
                                 unsigned int &_maxCellOcc)
    {
        for(auto &&f : _fluid)
        {
            ComputeMaxCellOccupancy(f, _solverProps, _maxCellOcc);
        }
    }

    //--------------------------------------------------------------------------------------

    void ComputeParticleVolume(std::shared_ptr<Rigid> _rigid,
                               std::shared_ptr<FluidSolverProperty> _solverProps);

    template<typename T>
    void ComputeParticleVolume(std::vector<T> _rigid,
                               std::shared_ptr<FluidSolverProperty> _solverProps)
    {
        for(auto &&r : _rigid)
        {
            ComputeParticleVolume(r, _solverProps);
        }
    }

    //--------------------------------------------------------------------------------------

    void ComputeDensity(std::shared_ptr<BaseSphParticle> _fluid,
                             std::shared_ptr<FluidSolverProperty> _solverProps,
                             const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<BaseSphParticle> _fluid,
                                  std::shared_ptr<BaseSphParticle> _fluidContributer,
                                  std::shared_ptr<FluidSolverProperty> _solverProps,
                                  const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<BaseSphParticle> _fluid,
                                  std::shared_ptr<Rigid> _rigid,
                                  std::shared_ptr<FluidSolverProperty> _solverProps,
                                  const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                        std::shared_ptr<FluidSolverProperty> _solverProps,
                        const bool accumulate = false);

    void ComputeDensity(std::shared_ptr<BaseSphParticle> _fluid,
                        std::vector<std::shared_ptr<Rigid>> _rigids,
                        std::shared_ptr<FluidSolverProperty> _solverProps,
                        const bool accumulate = false);

    template<typename T, typename ... Targs>
    void ComputeDensity(std::shared_ptr<BaseSphParticle> _fluid,
                        std::shared_ptr<FluidSolverProperty> _solverProps,
                        const bool accumulate,
                        T _fluidContributer,
                        Targs ... args)
    {
        ComputeDensity(_fluid, _fluidContributer, _solverProps, accumulate);
        ComputeDensity(_fluid, _solverProps, accumulate, args ...);
    }



    void ComputePressure(std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _solverProps);

//--------------------------------------------------------------------------------------

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _fluid
    void ComputePressureForce(std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<FluidSolverProperty> _solverProps,
                              const bool accumulate = false);

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _fluidContributer
    void ComputePressureForce(std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<BaseSphParticle> _fluidContributer,
                              std::shared_ptr<FluidSolverProperty> _solverProps,
                              const bool accumulate = false);

    /// @brief Method to compute pressure force exerted on _fluid body of fluid by _rigid
    void ComputePressureForce(std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<Rigid> _rigid,
                              std::shared_ptr<FluidSolverProperty> _solverProps,
                              const bool accumulate = false);

    void ComputePressureForce(std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                              std::shared_ptr<FluidSolverProperty> _solverProps,
                              const bool accumulate = false);

    void ComputePressureForce(std::shared_ptr<BaseSphParticle> _fluid,
                              std::vector<std::shared_ptr<Rigid>> _rigids,
                              std::shared_ptr<FluidSolverProperty> _solverProps,
                              const bool accumulate = false);

    template<typename T, typename ... Targs>
    void ComputePressureForce(std::shared_ptr<BaseSphParticle> _fluid,
                              std::shared_ptr<FluidSolverProperty> _solverProps,
                              const bool accumulate,
                              T _fluidContributer,
                              Targs ... args)
    {
        ComputePressureForce(_fluid, _fluidContributer, _solverProps, accumulate);
        ComputePressureForce(_fluid, _solverProps, accumulate, args ...);
    }


    //--------------------------------------------------------------------------------------

    void ComputeViscForce(std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<FluidSolverProperty> _solverProps);

    void ComputeViscForce(std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<Rigid> _rigid,
                          std::shared_ptr<FluidSolverProperty> _solverProps);

    void ComputeSurfaceTensionForce(std::shared_ptr<Fluid> _fluid,
                                    std::shared_ptr<FluidSolverProperty> _solverProps);

    void ComputeForces(std::shared_ptr<Fluid> _fluid,
                       std::shared_ptr<FluidSolverProperty> _solverProps,
                       const bool pressure = true,
                       const bool viscosity = true,
                       const bool surfTen = true,
                       const bool accumulate = false);

    void ComputeTotalForce(std::shared_ptr<Fluid> _fluid,
                           std::shared_ptr<FluidSolverProperty> _solverProps,
                           const bool accumulatePressure = true,
                           const bool accumulateViscous = true,
                           const bool accumulateSurfTen = true,
                           const bool accumulateExternal = true,
                           const bool accumulateGravity = true);

    //--------------------------------------------------------------------------------------

    void Integrate(std::shared_ptr<Fluid> _fluid,
                   std::shared_ptr<FluidSolverProperty> _solverProps);

    void HandleBoundaries(std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<FluidSolverProperty> _solverProps);



    //--------------------------------------------------------------------------------------
    // PCI SPH
    namespace pci
    {
        void PredictIntegrate(std::shared_ptr<Fluid> _fluid,
                              std::shared_ptr<FluidSolverProperty> _solverProps);

        void PredictDensity(std::shared_ptr<Fluid> _fluid,
                            std::shared_ptr<FluidSolverProperty> _solverProps);

        void predictDensityVariation(std::shared_ptr<Fluid> _fluid,
                                     std::shared_ptr<FluidSolverProperty> _solverProps);

        void ComputeMaxDensityVariation(std::shared_ptr<Fluid> _fluid,
                                        std::shared_ptr<FluidSolverProperty> _solverProps,
                                        float &_maxDenVar);

        void UpdatePressure(std::shared_ptr<Fluid> _fluid,
                            std::shared_ptr<FluidSolverProperty> _solverProps);

        void ComputePressureForce(std::shared_ptr<Fluid> _fluid,
                                  std::shared_ptr<FluidSolverProperty> _solverProps);


    }
}

#endif // SPH_H
