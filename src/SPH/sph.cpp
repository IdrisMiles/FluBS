#include "include/SPH/sph.h"
#include "SPH/gpudata.h"

void sph::ResetProperties(const FluidSolverProperty &_solverProps,
                          std::shared_ptr<BaseSphParticle> _sphParticles)
{
    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;

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
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               const bool accumulate)
{
    ParticleGpuData particleData = _fluid->GetParticleGpuData();

    sphGPU::ComputePressureForceFluid(particleData, _solverProps.gridResolution, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               std::shared_ptr<BaseSphParticle> _fluidContributer,
                               const bool accumulate)
{
    ParticleGpuData particle = _fluid->GetParticleGpuData();
    ParticleGpuData contributerParticle = _fluidContributer->GetParticleGpuData();

    sphGPU::ComputePressureForceFluidFluid(particle, contributerParticle, _solverProps.gridResolution, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputePressureForce(const FluidSolverProperty &_solverProps,
                               std::shared_ptr<BaseSphParticle> _fluid,
                               std::shared_ptr<Rigid> _rigid,
                               const bool accumulate)
{
    ParticleGpuData particle = _fluid->GetParticleGpuData();
    RigidGpuData rigidParticle = _rigid->GetRigidGpuData();

    sphGPU::ComputePressureForceFluidRigid(particle, rigidParticle, _solverProps.gridResolution, accumulate);
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
    FluidGpuData particleData = _fluid->GetFluidGpuData();

    sphGPU::ComputeViscForce(particleData, _solverProps.gridResolution);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeSurfaceTensionForce(const FluidSolverProperty &_solverProps,
                                     std::shared_ptr<Fluid> _fluid)
{
    FluidGpuData particleData = _fluid->GetFluidGpuData();

    sphGPU::ComputeSurfaceTensionForce(particleData, _solverProps.gridResolution);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeForces(const FluidSolverProperty &_solverProps,
                        std::shared_ptr<Fluid> _fluid,
                        const bool pressure,
                        const bool viscosity,
                        const bool surfTen,
                        const bool accumulate)
{
    FluidGpuData particleData = _fluid->GetFluidGpuData();

    sphGPU::ComputeForce(particleData, _solverProps.gridResolution, accumulate);
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
    FluidGpuData particleData = _fluid->GetFluidGpuData();

    sphGPU::ComputeTotalForce(particleData, _solverProps.gridResolution,
                              accumulatePressure,
                              accumulateViscous,
                              accumulateSurfTen,
                              accumulateExternal,
                              accumulateGravity);

}

//--------------------------------------------------------------------------------------------------------------------

void sph::Integrate(const FluidSolverProperty &_solverProps,
                    std::shared_ptr<BaseSphParticle> _particles)
{
    ParticleGpuData particleData = _particles->GetParticleGpuData();

    sphGPU::Integrate(particleData, _solverProps.gridResolution, _solverProps.deltaTime/_solverProps.solveIterations);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::HandleBoundaries(const FluidSolverProperty &_solverProps,
                           std::shared_ptr<BaseSphParticle> _fluid)
{
    ParticleGpuData particleData = _fluid->GetParticleGpuData();

    sphGPU::HandleBoundaries(particleData,
                             _solverProps.gridResolution,
                             (float)0.5f*_solverProps.gridCellWidth * _solverProps.gridResolution);
}


//--------------------------------------------------------------------------------------------------------------------
// Algae functions

void sph::ComputeAdvectionForce(const FluidSolverProperty &_solverProps,
                                std::shared_ptr<BaseSphParticle> _particles,
                                std::shared_ptr<Fluid> _advector,
                                const bool accumulate)
{
    ParticleGpuData particleData = _particles->GetParticleGpuData();
    FluidGpuData advectorParticleData = _advector->GetFluidGpuData();

    sphGPU::ComputeAdvectionForce(particleData, advectorParticleData, _solverProps.gridResolution, accumulate);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::AdvectParticle(const FluidSolverProperty &_solverProps,
                         std::shared_ptr<BaseSphParticle> _particles,
                         std::shared_ptr<Fluid> _advector)
{
    ParticleGpuData particleData = _particles->GetParticleGpuData();
    FluidGpuData advectorParticleData = _advector->GetFluidGpuData();

    sphGPU::AdvectParticle(particleData,
                           advectorParticleData,
                           _solverProps.gridResolution,
                           _solverProps.deltaTime/_solverProps.solveIterations);
}

//--------------------------------------------------------------------------------------------------------------------

void sph::ComputeBioluminescence(const FluidSolverProperty &_solverProps,
                                 std::shared_ptr<Algae> _algae)
{
    AlgaeGpuData particleData = _algae->GetAlgaeGpuData();

    sphGPU::ComputeBioluminescence(particleData, _solverProps.gridResolution, _solverProps.deltaTime/_solverProps.solveIterations);
}

