#include "include/SPH/sph.h"
#include "SPH/gpudata.h"

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<BaseSphParticle> _sphParticles)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;
    auto fluidProps =  _sphParticles->GetProperty();

    ParticleGpuData particle = _sphParticles->GetParticleGpuData();

    sphGPU::ResetProperties(particle, numCells);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::vector<std::shared_ptr<BaseSphParticle>> _sphParticles)
{
    for(auto &&sp : _sphParticles)
    {
        ResetProperties(_solverProps, sp);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Fluid> _fluid)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;

    FluidGpuData particle = _fluid->GetFluidGpuData();

    sphGPU::ResetProperties(particle, numCells);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::vector<std::shared_ptr<Fluid>> _fluid)
{
    for(auto &&f : _fluid)
    {
        ResetProperties(_solverProps, f);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Rigid> _rigid)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;

    ParticleGpuData particle = _rigid->GetParticleGpuData();

    sphGPU::ResetProperties(particle, numCells);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::vector<std::shared_ptr<Rigid>> _rigid)
{
    for(auto &&r : _rigid)
    {
        ResetProperties(_solverProps, r);
    }
}

//--------------------------------------------------------------------------------------------------------------------
void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Algae> _algae)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;

    ParticleGpuData particle = _algae->GetParticleGpuData();

    sphGPU::ResetProperties(particle, numCells);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ResetTotalForce(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<BaseSphParticle> _sphParticles)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;
    auto fluidProps =  _sphParticles->GetProperty();

    sphGPU::ResetTotalForce(_sphParticles->GetTotalForcePtr(), fluidProps->numParticles);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::InitFluidAsCube(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<BaseSphParticle> _sphParticles)
{
    ParticleGpuData particle = _sphParticles->GetParticleGpuData();
    auto fluidProps =  _sphParticles->GetProperty();
    sphGPU::InitFluidAsCube(particle, ceil(cbrt(fluidProps->numParticles)), 2.0f*fluidProps->particleRadius);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::InitAlgaeIllumination(const FluidSolverProperty &_solverProps, std::shared_ptr<Algae> _algae)
{
    auto algaeProps = _algae->GetProperty();
    sphGPU::InitAlgaeIllumination(_algae->GetIlluminationPtr(), algaeProps->numParticles);
    sphGPU::InitAlgaeIllumination(_algae->GetPrevPressurePtr(), algaeProps->numParticles);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::InitSphParticleIds(std::shared_ptr<BaseSphParticle> _sphParticles)
{
    sphGPU::InitSphParticleIds(_sphParticles->GetParticleIdPtr(), _sphParticles->GetProperty()->numParticles);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeHash(const FluidSolverProperty &_solverProps,
                      std::shared_ptr<BaseSphParticle> _sphParticles)
{

    ParticleGpuData particle = _sphParticles->GetParticleGpuData();

    sphGPU::ParticleHash(particle, _solverProps.gridResolution, _solverProps.gridCellWidth);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::SortParticlesByHash(std::shared_ptr<BaseSphParticle> _sphParticles)
{
    ParticleGpuData particle = _sphParticles->GetParticleGpuData();
    sphGPU::SortParticlesByHash(particle);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::SortParticlesByHash(std::shared_ptr<Algae> _sphParticles)
{
    AlgaeGpuData particle = _sphParticles->GetAlgaeGpuData();
    sphGPU::SortParticlesByHash(particle);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeParticleScatterIds(const FluidSolverProperty &_solverProps,
                                    std::shared_ptr<BaseSphParticle> _sphParticles)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;

    sphGPU::ComputeParticleScatterIds(_sphParticles->GetCellOccupancyPtr(), _sphParticles->GetCellParticleIdxPtr(), numCells);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeMaxCellOccupancy(const FluidSolverProperty &_solverProps,
                                  std::shared_ptr<BaseSphParticle> _sphParticles,
                                  unsigned int &_maxCellOcc)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;

    sphGPU::ComputeMaxCellOccupancy(_sphParticles->GetCellOccupancyPtr(), numCells, _maxCellOcc);

    _sphParticles->SetMaxCellOcc(_maxCellOcc);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeParticleVolume(const FluidSolverProperty &_solverProps,
                                std::shared_ptr<Rigid> _rigid)
{
    RigidGpuData particle = _rigid->GetRigidGpuData();

    sphGPU::ComputeParticleVolume(particle, _solverProps.gridResolution);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeParticleVolume(const FluidSolverProperty &_solverProps,
                                std::vector<std::shared_ptr<Rigid>> _rigid)
{
    for(auto &&r : _rigid)
    {
        ComputeParticleVolume(_solverProps, r);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeDensity(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _fluid,
                         const bool accumulate)
{
    auto fluidProps =  _fluid->GetProperty();
    ParticleGpuData particle = _fluid->GetParticleGpuData();

    sphGPU::ComputeDensity(particle, _solverProps.gridResolution, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeDensity(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _fluid,
                         std::shared_ptr<BaseSphParticle> _fluidContributer,
                         const bool accumulate)
{
    ParticleGpuData fluidParticleData = _fluid->GetParticleGpuData();
    ParticleGpuData fluidContributerParticleData = _fluidContributer->GetParticleGpuData();

    sphGPU::ComputeDensityFluidFluid( fluidParticleData, fluidContributerParticleData, _solverProps.gridResolution, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeDensity(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _fluid,
                         std::shared_ptr<Rigid> _rigid,
                         const bool accumulate)
{
    ParticleGpuData fluidParticleData = _fluid->GetParticleGpuData();
    RigidGpuData rigidParticleData = _rigid->GetRigidGpuData();

    sphGPU::ComputeDensityFluidRigid(fluidParticleData, rigidParticleData, _solverProps.gridResolution, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeDensity(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _fluid,
                         std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                         const bool accumulate)
{
    for(auto &&fc : _fluidContributers)
    {
        sph::ComputeDensity(_solverProps, _fluid, fc, accumulate);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeDensity(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _fluid,
                                    std::vector<std::shared_ptr<Rigid>> _rigids,
                                    const bool accumulate)
{
    for(auto &&r : _rigids)
    {
        sph::ComputeDensity(_solverProps, _fluid, r, accumulate);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressure(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<BaseSphParticle> _particles)
{
    auto particleProps =  _particles->GetProperty();

//    sphGPU::ComputePressureFluid(_fluid->GetMaxCellOcc(),
//                            _solverProps.gridResolution,
//                            _fluid->GetPressurePtr(),
//                            _fluid->GetDensityPtr(),
//                            particleProps->restDensity,
//                            particleProps->gasStiffness,
//                            _fluid->GetCellOccupancyPtr(),
//                            _fluid->GetCellParticleIdxPtr(),
//                            particleProps->numParticles);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressure(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Fluid> _fluid)
{
    FluidGpuData particle = _fluid->GetFluidGpuData();

    sphGPU::ComputePressureFluid(particle, _solverProps.gridResolution);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressure(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<Algae> _algae,
                          std::shared_ptr<Fluid> _fluid)
{
    ParticleGpuData particleData = _algae->GetParticleGpuData();
    ParticleGpuData contributerParticleData = _fluid->GetParticleGpuData();

    sphGPU::SamplePressure(particleData, contributerParticleData, _solverProps.gridResolution);

//    sphGPU::ComputePressureFluid(_fluid->GetMaxCellOcc(),
//                                 _solverProps.gridResolution,
//                                 _algae->GetPrevPressurePtr()
//                                 _algae->GetPressurePtr(),
//                                 _algae->GetDensityPtr(),
//                                 algaeProps->restDensity,
////                                 algaeProps->gasStiffness,
//                                 _algae->GetCellOccupancyPtr(),
//                                 _algae->GetCellParticleIdxPtr(),
//                                 algaeProps->numParticles);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               const bool accumulate)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::ComputePressureForceFluid(_fluid->GetMaxCellOcc(),
                                 _solverProps.gridResolution,
                                 _fluid->GetPressureForcePtr(),
                                 _fluid->GetPressurePtr(),
                                 _fluid->GetDensityPtr(),
                                 fluidProps->particleMass,
                                 _fluid->GetPositionPtr(),
                                 _fluid->GetCellOccupancyPtr(),
                                 _fluid->GetCellParticleIdxPtr(),
                                 fluidProps->numParticles,
                                 fluidProps->smoothingLength, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               std::shared_ptr<BaseSphParticle> _fluidContributer,
                               const bool accumulate)
{
    auto fluidProps = _fluid->GetProperty();
    auto fluidContribProps = _fluidContributer->GetProperty();

    sphGPU::ComputePressureForceFluidFluid(_fluid->GetMaxCellOcc(),
                                 _solverProps.gridResolution,
                                 _fluid->GetPressureForcePtr(),
                                 _fluid->GetPressurePtr(),
                                 _fluid->GetDensityPtr(),
                                 fluidProps->particleMass,
                                 _fluid->GetPositionPtr(),
                                 _fluid->GetCellOccupancyPtr(),
                                 _fluid->GetCellParticleIdxPtr(),
                                 _fluidContributer->GetPressurePtr(),
                                 _fluidContributer->GetDensityPtr(),
                                 fluidContribProps->particleMass,
                                 _fluidContributer->GetPositionPtr(),
                                 _fluidContributer->GetCellOccupancyPtr(),
                                 _fluidContributer->GetCellParticleIdxPtr(),
                                 fluidProps->numParticles,
                                 fluidProps->smoothingLength, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               std::shared_ptr<Rigid> _rigid,
                               const bool accumulate)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::ComputePressureForceFluidRigid(_fluid->GetMaxCellOcc(),
                                           _solverProps.gridResolution,
                                           _fluid->GetPressureForcePtr(),
                                           _fluid->GetPressurePtr(),
                                           _fluid->GetDensityPtr(),
                                           fluidProps->particleMass,
                                           _fluid->GetPositionPtr(),
                                           _fluid->GetCellOccupancyPtr(),
                                           _fluid->GetCellParticleIdxPtr(),
                                           fluidProps->restDensity,
                                           _rigid->GetVolumePtr(),
                                           _rigid->GetPositionPtr(),
                                           _rigid->GetCellOccupancyPtr(),
                                           _rigid->GetCellParticleIdxPtr(),
                                           fluidProps->numParticles,
                                           fluidProps->smoothingLength, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               std::vector<std::shared_ptr<BaseSphParticle>> &_fluidContributers,
                               const bool accumulate)
{
    for(auto &&fc : _fluidContributers)
    {
        sph::ComputePressureForce(_solverProps, _fluid, fc, accumulate);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               std::vector<std::shared_ptr<Rigid>> _rigids,
                               const bool accumulate)
{
    for(auto &&r : _rigids)
    {
        sph::ComputePressureForce(_solverProps, _fluid, r, accumulate);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeViscForce(const FluidSolverProperty &_solverProps,
                           std::shared_ptr<Fluid> _fluid)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::ComputeViscForce(_fluid->GetMaxCellOcc(),
                             _solverProps.gridResolution,
                             _fluid->GetViscForcePtr(),
                             fluidProps->viscosity,
                             _fluid->GetVelocityPtr(),
                             _fluid->GetDensityPtr(),
                             fluidProps->particleMass,
                             _fluid->GetPositionPtr(),
                             _fluid->GetCellOccupancyPtr(),
                             _fluid->GetCellParticleIdxPtr(),
                             fluidProps->numParticles,
                             fluidProps->smoothingLength);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeSurfaceTensionForce(const FluidSolverProperty &_solverProps,
                                     std::shared_ptr<Fluid> _fluid)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::ComputeSurfaceTensionForce(_fluid->GetMaxCellOcc(),
                                       _solverProps.gridResolution,
                                       _fluid->GetSurfTenForcePtr(),
                                       fluidProps->surfaceTension,
                                       fluidProps->surfaceThreshold,
                                       _fluid->GetDensityPtr(),
                                       fluidProps->particleMass,
                                       _fluid->GetPositionPtr(),
                                       _fluid->GetCellOccupancyPtr(),
                                       _fluid->GetCellParticleIdxPtr(),
                                       fluidProps->numParticles,
                                       fluidProps->smoothingLength);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeForces(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<Fluid> _fluid,
                        const bool pressure,
                        const bool viscosity,
                        const bool surfTen,
                        const bool accumulate)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::ComputeForce(_fluid->GetMaxCellOcc(),
                         _solverProps.gridResolution,
                         _fluid->GetPressureForcePtr(),
                         _fluid->GetViscForcePtr(),
                         _fluid->GetSurfTenForcePtr(),
                         fluidProps->viscosity,
                         fluidProps->surfaceTension,
                         fluidProps->surfaceThreshold,
                         _fluid->GetPressurePtr(),
                         _fluid->GetDensityPtr(),
                         fluidProps->particleMass,
                         _fluid->GetPositionPtr(),
                         _fluid->GetVelocityPtr(),
                         _fluid->GetCellOccupancyPtr(),
                         _fluid->GetCellParticleIdxPtr(),
                         fluidProps->numParticles,
                         fluidProps->smoothingLength, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeTotalForce(const FluidSolverProperty &_solverProps,
                            std::shared_ptr<Fluid> _fluid,
                            const bool accumulatePressure,
                            const bool accumulateViscous,
                            const bool accumulateSurfTen,
                            const bool accumulateExternal,
                            const bool accumulateGravity)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::ComputeTotalForce(_fluid->GetMaxCellOcc(),
                              _solverProps.gridResolution,
                              accumulatePressure,
                              accumulateViscous,
                              accumulateSurfTen,
                              accumulateExternal,
                              accumulateGravity,
                              _fluid->GetTotalForcePtr(),
                              _fluid->GetExternalForcePtr(),
                              _fluid->GetPressureForcePtr(),
                              _fluid->GetViscForcePtr(),
                              _fluid->GetSurfTenForcePtr(),
                              fluidProps->gravity,
                              fluidProps->particleMass,
                              _fluid->GetPositionPtr(),
                              _fluid->GetVelocityPtr(),
                              _fluid->GetCellOccupancyPtr(),
                              _fluid->GetCellParticleIdxPtr(),
                              fluidProps->numParticles,
                              fluidProps->smoothingLength);

}

//--------------------------------------------------------------------------------------------------------------------

void sph::Integrate(const FluidSolverProperty &_solverProps,
                    std::shared_ptr<BaseSphParticle> _particles)
{
    auto fluidProps = _particles->GetProperty();

    sphGPU::Integrate(_particles->GetMaxCellOcc(),
                      _solverProps.gridResolution,
                      _particles->GetTotalForcePtr(),
                      _particles->GetPositionPtr(),
                      _particles->GetVelocityPtr(),
                      _solverProps.deltaTime/_solverProps.solveIterations,
                      fluidProps->numParticles);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::HandleBoundaries(const FluidSolverProperty &_solverProps,
                           std::shared_ptr<BaseSphParticle> _fluid)
{
    auto fluidProps = _fluid->GetProperty();

    sphGPU::HandleBoundaries(_fluid->GetMaxCellOcc(),
                             _solverProps.gridResolution,
                             _fluid->GetPositionPtr(),
                             _fluid->GetVelocityPtr(),
                             (float)0.5f*_solverProps.gridCellWidth * _solverProps.gridResolution,
                             fluidProps->numParticles);
}


//--------------------------------------------------------------------------------------------------------------------
// Algae functions

void sph::ComputeAdvectionForce(const FluidSolverProperty &_solverProps,
                                std::shared_ptr<BaseSphParticle> _particles,
                                std::shared_ptr<Fluid> _advector,
                                const bool accumulate)
{
    auto particleProps = _particles->GetProperty();
    auto advectorProps = _advector->GetProperty();

    sphGPU::ComputeAdvectionForce(_particles->GetMaxCellOcc(),
                                  _solverProps.gridResolution,
                                  _particles->GetPositionPtr(),
                                  _particles->GetVelocityPtr(),
                                  _particles->GetTotalForcePtr(),
                                  _particles->GetCellOccupancyPtr(),
                                  _particles->GetCellParticleIdxPtr(),
                                  _advector->GetPositionPtr(),
                                  _advector->GetTotalForcePtr(),
                                  _advector->GetDensityPtr(),
                                  advectorProps->particleMass,
                                  _advector->GetCellOccupancyPtr(),
                                  _advector->GetCellParticleIdxPtr(),
                                  particleProps->numParticles,
                                  particleProps->smoothingLength,
                                  accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::AdvectParticle(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _particles,
                         std::shared_ptr<Fluid> _advector)
{
    auto particleProps = _particles->GetProperty();
    auto advectorProps = _advector->GetProperty();

    sphGPU::AdvectParticle(_particles->GetMaxCellOcc(),
                           _solverProps.gridResolution,
                           _particles->GetPositionPtr(),
                           _particles->GetVelocityPtr(),
                           _particles->GetCellOccupancyPtr(),
                           _particles->GetCellParticleIdxPtr(),
                           _advector->GetPositionPtr(),
                           _advector->GetVelocityPtr(),
                           _advector->GetDensityPtr(),
                           advectorProps->particleMass,
                           _advector->GetCellOccupancyPtr(),
                           _advector->GetCellParticleIdxPtr(),
                           particleProps->numParticles,
                           particleProps->smoothingLength,
                           _solverProps.deltaTime/_solverProps.solveIterations);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeBioluminescence(const FluidSolverProperty &_solverProps,
                                 std::shared_ptr<Algae> _algae)
{
    auto algaeProps = _algae->GetProperty();

    sphGPU::ComputeBioluminescence(_algae->GetMaxCellOcc(),
                                   _solverProps.gridResolution,
                                   _algae->GetPressurePtr(),
                                   _algae->GetPrevPressurePtr(),
                                   _algae->GetIlluminationPtr(),
                                   algaeProps->bioluminescenceThreshold,
                                   algaeProps->reactionRate,
                                   algaeProps->deactionRate,
                                   _solverProps.deltaTime/_solverProps.solveIterations,
                                   algaeProps->numParticles);
}




//--------------------------------------------------------------------------------------------------------------------
// PCISPH functions

void sph::pci::PredictIntegrate(const FluidSolverProperty &_solverProps,
                                std::shared_ptr<Fluid> _fluid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void sph::pci::PredictDensity(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<Fluid> _fluid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void sph::pci::predictDensityVariation(const FluidSolverProperty &_solverProps,
                                       std::shared_ptr<Fluid> _fluid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void sph::pci::ComputeMaxDensityVariation(const FluidSolverProperty &_solverProps,
                                          std::shared_ptr<Fluid> _fluid,
                                          float &_maxDenVar)
{

}

//--------------------------------------------------------------------------------------------------------------------

void sph::pci::UpdatePressure(const FluidSolverProperty &_solverProps,
                              std::shared_ptr<Fluid> _fluid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void sph::pci::ComputePressureForce(const FluidSolverProperty &_solverProps,
                                    std::shared_ptr<Fluid> _fluid)
{

}
