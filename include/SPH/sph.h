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

    void ResetTotalForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    void InitFluidAsCube(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<BaseSphParticle> _sphParticles);

    //--------------------------------------------------------------------------------------
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

    void SortParticlesByHash(std::shared_ptr<BaseSphParticle> _sphParticles);

    template<typename T>
    void SortParticlesByHash(std::vector<T> _sphParticles)
    {
        for(auto &&f : _sphParticles)
        {
            SortParticlesByHash(f);
        }
    }

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

    //--------------------------------------------------------------------------------------

    void ComputeParticleVolume(std::shared_ptr<FluidSolverProperty> _solverProps,
                               std::shared_ptr<Rigid> _rigid);

    void ComputeParticleVolume(std::shared_ptr<FluidSolverProperty> _solverProps,
                               std::vector<std::shared_ptr<Rigid>> _rigid);

    //--------------------------------------------------------------------------------------

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



    void ComputePressure(std::shared_ptr<FluidSolverProperty> _solverProps,
                         std::shared_ptr<Fluid> _fluid);

//--------------------------------------------------------------------------------------

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


    //--------------------------------------------------------------------------------------

    void ComputeViscForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                          std::shared_ptr<Fluid> _fluid);

    void ComputeViscForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                          std::shared_ptr<Fluid> _fluid,
                          std::shared_ptr<Rigid> _rigid);

    void ComputeSurfaceTensionForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                                    std::shared_ptr<Fluid> _fluid);

    void ComputeForces(std::shared_ptr<FluidSolverProperty> _solverProps,
                       std::shared_ptr<Fluid> _fluid,
                       const bool pressure = true,
                       const bool viscosity = true,
                       const bool surfTen = true,
                       const bool accumulate = false);

    void ComputeTotalForce(std::shared_ptr<FluidSolverProperty> _solverProps,
                           std::shared_ptr<Fluid> _fluid,
                           const bool accumulatePressure = true,
                           const bool accumulateViscous = true,
                           const bool accumulateSurfTen = true,
                           const bool accumulateExternal = true,
                           const bool accumulateGravity = true);

    //--------------------------------------------------------------------------------------

    void Integrate(std::shared_ptr<FluidSolverProperty> _solverProps,
                   std::shared_ptr<Fluid> _fluid);

    void HandleBoundaries(std::shared_ptr<FluidSolverProperty> _solverProps,
                          std::shared_ptr<Fluid> _fluid);



    //--------------------------------------------------------------------------------------
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
