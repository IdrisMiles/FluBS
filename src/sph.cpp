#include "include/sph.h"

//void sph::InitFluidAsCube(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps =  _fluid->GetFluidProperty();

//    sphGPU::InitFluidAsCube(_fluid->GetPositionPtr(),
//                            _fluid->GetVelocityPtr(),
//                            _fluid->GetDensityPtr(),
//                            fluidProps->restDensity,
//                            fluidProps->numParticles,
//                            ceil(cbrt(fluidProps->numParticles)),
//                            2.0f*fluidProps->particleRadius);
//}

//void sph::ComputeHash(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps =  _fluid->GetFluidProperty();

//    sphGPU::ParticleHash(_fluid->GetParticleHashIdPtr(),
//                         _fluid->GetCellOccupancyPtr(),
//                         _fluid->GetPositionPtr(),
//                         fluidProps->numParticles,
//                         _solverProps->gridResolution,
//                         _solverProps->gridCellWidth);
//}

//void sph::ComputePressure(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps =  _fluid->GetFluidProperty();

//    sphGPU::ComputePressure(_fluid->GetMaxCellOcc(),
//                            _solverProps->gridResolution,
//                            _fluid->GetPressurePtr(),
//                            _fluid->GetDensityPtr(),
//                            fluidProps->restDensity,
//                            fluidProps->gasStiffness,
//                            _fluid->GetMassPtr(),
//                            _fluid->GetCellOccupancyPtr(),
//                            _fluid->GetCellParticleIdxPtr(),
//                            _fluid->GetPositionPtr(),
//                            fluidProps->numParticles,
//                            fluidProps->smoothingLength);
//}

//void sph::ComputePressureForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps = _fluid->GetFluidProperty();

//    sphGPU::ComputePressureForce(_fluid->GetMaxCellOcc(),
//                                 _solverProps->gridResolution,
//                                 _fluid->GetPressureForcePtr(),
//                                 _fluid->GetPressurePtr(),
//                                 _fluid->GetDensityPtr(),
//                                 _fluid->GetMassPtr(),
//                                 _fluid->GetPositionPtr(),
//                                 _fluid->GetCellOccupancyPtr(),
//                                 _fluid->GetCellParticleIdxPtr(),
//                                 fluidProps->numParticles,
//                                 fluidProps->smoothingLength);
//}

//void sph::ComputeViscForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps = _fluid->GetFluidProperty();

//    sphGPU::ComputeViscForce(_fluid->GetMaxCellOcc(),
//                             _solverProps->gridResolution,
//                             _fluid->GetViscForcePtr(),
//                             fluidProps->viscosity,
//                             _fluid->GetVelocityPtr(),
//                             _fluid->GetDensityPtr(),
//                             _fluid->GetMassPtr(),
//                             _fluid->GetPositionPtr(),
//                             _fluid->GetCellOccupancyPtr(),
//                             _fluid->GetCellParticleIdxPtr(),
//                             fluidProps->numParticles,
//                             fluidProps->smoothingLength);
//}

//void sph::ComputeSurfaceTensionForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps = _fluid->GetFluidProperty();

//    sphGPU::ComputeSurfaceTensionForce(_fluid->GetMaxCellOcc(),
//                                       _solverProps->gridResolution,
//                                       _fluid->GetSurfTenForcePtr(),
//                                       fluidProps->surfaceTension,
//                                       fluidProps->surfaceThreshold,
//                                       _fluid->GetDensityPtr(),
//                                       _fluid->GetMassPtr(),
//                                       _fluid->GetPositionPtr(),
//                                       _fluid->GetCellOccupancyPtr(),
//                                       _fluid->GetCellParticleIdxPtr(),
//                                       fluidProps->numParticles,
//                                       fluidProps->smoothingLength);
//}

//void sph::ComputeTotalForce(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps = _fluid->GetFluidProperty();

//    sphGPU::ComputeTotalForce(_fluid->GetMaxCellOcc(),
//                              _solverProps->gridResolution,
//                              _fluid->GetTotalForcePtr(),
//                              _fluid->GetExternalForcePtr(),
//                              _fluid->GetPressureForcePtr(),
//                              _fluid->GetViscForcePtr(),
//                              _fluid->GetSurfTenForcePtr(),
//                              _fluid->GetMassPtr(),
//                              _fluid->GetPositionPtr(),
//                              _fluid->GetVelocityPtr(),
//                              _fluid->GetCellOccupancyPtr(),
//                              _fluid->GetCellParticleIdxPtr(),
//                              fluidProps->numParticles,
//                              fluidProps->smoothingLength);
//}

//void sph::Integrate(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps = _fluid->GetFluidProperty();

//    sphGPU::Integrate(_fluid->GetMaxCellOcc(),
//                      _solverProps->gridResolution,
//                      _fluid->GetTotalForcePtr(),
//                      _fluid->GetPositionPtr(),
//                      _fluid->GetVelocityPtr(),
//                      _solverProps->deltaTime,
//                      fluidProps->numParticles);
//}

//void sph::HandleBoundaries(std::shared_ptr<Fluid> _fluid, std::shared_ptr<FluidSolverProperty> _solverProps)
//{
//    auto fluidProps = _fluid->GetFluidProperty();

//    sphGPU::HandleBoundaries(_fluid->GetMaxCellOcc(),
//                             _solverProps->gridResolution,
//                             _fluid->GetPositionPtr(),
//                             _fluid->GetVelocityPtr(),
//                             _solverProps->gridResolution,
//                             fluidProps->numParticles);
//}
