#include "include/SPH/sph.h"

void sph::ResetProperties(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    auto fluidProps =  _fluid->GetFluidProperty();

    sphGPU::ResetProperties(_fluid->GetPressureForcePtr(),
                            _fluid->GetViscForcePtr(),
                            _fluid->GetSurfTenForcePtr(),
                            _fluid->GetExternalForcePtr(),
                            _fluid->GetTotalForcePtr(),
                            _fluid->GetMassPtr(),
                            _fluid->GetDensityPtr(),
                            _fluid->GetPressurePtr(),
                            _fluid->GetParticleHashIdPtr(),
                            _fluid->GetCellOccupancyPtr(),
                            _fluid->GetCellParticleIdxPtr(),
                            fluidProps->particleMass,
                            numCells,
                            fluidProps->numParticles);
}

void sph::InitFluidAsCube(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps =  _fluid->GetFluidProperty();

    sphGPU::InitFluidAsCube(_fluid->GetPositionPtr(),
                            _fluid->GetVelocityPtr(),
                            _fluid->GetDensityPtr(),
                            fluidProps->restDensity,
                            fluidProps->numParticles,
                            ceil(cbrt(fluidProps->numParticles)),
                            2.0f*fluidProps->particleRadius);
}

void sph::ComputeHash(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps =  _fluid->GetFluidProperty();

    sphGPU::ParticleHash(_fluid->GetParticleHashIdPtr(),
                         _fluid->GetCellOccupancyPtr(),
                         _fluid->GetPositionPtr(),
                         fluidProps->numParticles,
                         _solverProps->gridResolution,
                         _solverProps->gridCellWidth);
}

void sph::SortParticlesByHash(std::shared_ptr<Fluid> _fluid)
{
    sphGPU::SortParticlesByHash(_fluid->GetParticleHashIdPtr(),
                                _fluid->GetPositionPtr(),
                                _fluid->GetVelocityPtr(),
                                _fluid->GetFluidProperty()->numParticles);
}

void sph::ComputeParticleScatterIds(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;

    sphGPU::ComputeParticleScatterIds(_fluid->GetCellOccupancyPtr(),
                                      _fluid->GetCellParticleIdxPtr(),
                                      numCells);
}

void sph::ComputeMaxCellOccupancy(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps, unsigned int &_maxCellOcc)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;

    sphGPU::ComputeMaxCellOccupancy(_fluid->GetCellOccupancyPtr(),
                                    numCells,
                                    _maxCellOcc);

    _fluid->SetMaxCellOcc(_maxCellOcc);
}

void sph::ComputeParticleVolume(std::shared_ptr<Boundary> _boundary, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps =  _boundary->GetFluidProperty();

    sphGPU::ComputeParticleVolume(_boundary->GetMaxCellOcc(),
                                  _solverProps->gridResolution,
                                  _boundary->GetVolumePtr(),
                                  _boundary->GetCellOccupancyPtr(),
                                  _boundary->GetCellParticleIdxPtr(),
                                  _boundary->GetPositionPtr(),
                                  fluidProps->numParticles,
                                  fluidProps->smoothingLength);
}

void sph::ComputeDensityFluid(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps, const bool accumulate)
{
    auto fluidProps =  _fluid->GetFluidProperty();

    sphGPU::ComputeDensity(_fluid->GetMaxCellOcc(),
                            _solverProps->gridResolution,
                            _fluid->GetDensityPtr(),
                            _fluid->GetMassPtr(),
                            _fluid->GetCellOccupancyPtr(),
                            _fluid->GetCellParticleIdxPtr(),
                            _fluid->GetPositionPtr(),
                            fluidProps->numParticles,
                            fluidProps->smoothingLength,
                            accumulate);
}

void sph::ComputeDensityFluidFluid(std::shared_ptr<Fluid> _fluid,
                                   std::shared_ptr<Fluid> _fluidContributer,
                                   std::shared_ptr<FluidSolverProperty> _solverProps,
                                   const bool accumulate)
{
//    auto fluidProps =  _fluid->GetFluidProperty();

//    sphGPU::ComputeDensityFluidFluid(_fluid->GetMaxCellOcc(),
//                                     _solverProps->gridResolution,
//                                     _fluid->GetDensityPtr(),
//                                     _fluid->GetPositionPtr(),
//                                     _fluid->GetCellOccupancyPtr(),
//                                     _fluid->GetCellParticleIdxPtr(),
//                                     _fluidContributer->GetMassPtr(),
//                                     _fluidContributer->GetPositionPtr(),
//                                     _fluidContributer->GetCellOccupancyPtr(),
//                                     _fluidContributer->GetCellParticleIdxPtr(),
//                                     fluidProps->numParticles,
//                                     fluidProps->smoothingLength,
//                                     accumulate);
}

void sph::ComputeDensityFluidBoundary(std::shared_ptr<Fluid> _fluid,
                                      std::shared_ptr<Boundary> _boundary,
                                      std::shared_ptr<FluidSolverProperty> _solverProps,
                                      const bool accumulate)
{
    auto fluidProps =  _fluid->GetFluidProperty();
    auto boundaryProps =  _boundary->GetFluidProperty();

    sphGPU::ComputeDensityFluidBoundary(_fluid->GetMaxCellOcc(),
                                        _solverProps->gridResolution,
                                        fluidProps->numParticles,
                                        fluidProps->restDensity,
                                        _fluid->GetDensityPtr(),
                                        _fluid->GetPositionPtr(),
                                        _fluid->GetCellOccupancyPtr(),
                                        _fluid->GetCellParticleIdxPtr(),
                                        _boundary->GetVolumePtr(),
                                        _boundary->GetPositionPtr(),
                                        _boundary->GetCellOccupancyPtr(),
                                        _boundary->GetCellParticleIdxPtr(),
                                        fluidProps->smoothingLength,
                                        accumulate);
}


void sph::ComputePressure(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps =  _fluid->GetFluidProperty();

    sphGPU::ComputePressure(_fluid->GetMaxCellOcc(),
                            _solverProps->gridResolution,
                            _fluid->GetPressurePtr(),
                            _fluid->GetDensityPtr(),
                            fluidProps->restDensity,
                            fluidProps->gasStiffness,
                            _fluid->GetCellOccupancyPtr(),
                            _fluid->GetCellParticleIdxPtr(),
                            fluidProps->numParticles);
}

void sph::ComputePressureForceFluid(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps, const bool accumulate)
{
    auto fluidProps = _fluid->GetFluidProperty();

    sphGPU::ComputePressureForce(_fluid->GetMaxCellOcc(),
                                 _solverProps->gridResolution,
                                 _fluid->GetPressureForcePtr(),
                                 _fluid->GetPressurePtr(),
                                 _fluid->GetDensityPtr(),
                                 _fluid->GetMassPtr(),
                                 _fluid->GetPositionPtr(),
                                 _fluid->GetCellOccupancyPtr(),
                                 _fluid->GetCellParticleIdxPtr(),
                                 fluidProps->numParticles,
                                 fluidProps->smoothingLength, accumulate);
}

void sph::ComputeViscForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps = _fluid->GetFluidProperty();

    sphGPU::ComputeViscForce(_fluid->GetMaxCellOcc(),
                             _solverProps->gridResolution,
                             _fluid->GetViscForcePtr(),
                             fluidProps->viscosity,
                             _fluid->GetVelocityPtr(),
                             _fluid->GetDensityPtr(),
                             _fluid->GetMassPtr(),
                             _fluid->GetPositionPtr(),
                             _fluid->GetCellOccupancyPtr(),
                             _fluid->GetCellParticleIdxPtr(),
                             fluidProps->numParticles,
                             fluidProps->smoothingLength);
}

void sph::ComputeSurfaceTensionForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps = _fluid->GetFluidProperty();

    sphGPU::ComputeSurfaceTensionForce(_fluid->GetMaxCellOcc(),
                                       _solverProps->gridResolution,
                                       _fluid->GetSurfTenForcePtr(),
                                       fluidProps->surfaceTension,
                                       fluidProps->surfaceThreshold,
                                       _fluid->GetDensityPtr(),
                                       _fluid->GetMassPtr(),
                                       _fluid->GetPositionPtr(),
                                       _fluid->GetCellOccupancyPtr(),
                                       _fluid->GetCellParticleIdxPtr(),
                                       fluidProps->numParticles,
                                       fluidProps->smoothingLength);
}

void sph::ComputeTotalForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps = _fluid->GetFluidProperty();

    sphGPU::ComputeTotalForce(_fluid->GetMaxCellOcc(),
                              _solverProps->gridResolution,
                              _fluid->GetTotalForcePtr(),
                              _fluid->GetExternalForcePtr(),
                              _fluid->GetPressureForcePtr(),
                              _fluid->GetViscForcePtr(),
                              _fluid->GetSurfTenForcePtr(),
                              _fluid->GetMassPtr(),
                              _fluid->GetPositionPtr(),
                              _fluid->GetVelocityPtr(),
                              _fluid->GetCellOccupancyPtr(),
                              _fluid->GetCellParticleIdxPtr(),
                              fluidProps->numParticles,
                              fluidProps->smoothingLength);

}

void sph::Integrate(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps = _fluid->GetFluidProperty();

    sphGPU::Integrate(_fluid->GetMaxCellOcc(),
                      _solverProps->gridResolution,
                      _fluid->GetTotalForcePtr(),
                      _fluid->GetPositionPtr(),
                      _fluid->GetVelocityPtr(),
                      _solverProps->deltaTime,
                      fluidProps->numParticles);
}

void sph::HandleBoundaries(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
{
    auto fluidProps = _fluid->GetFluidProperty();

    sphGPU::HandleBoundaries(_fluid->GetMaxCellOcc(),
                             _solverProps->gridResolution,
                             _fluid->GetPositionPtr(),
                             _fluid->GetVelocityPtr(),
                             (float)0.5f*_solverProps->gridCellWidth * _solverProps->gridResolution,
                             fluidProps->numParticles);
}
