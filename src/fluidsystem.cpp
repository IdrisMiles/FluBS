#include "include/fluidsystem.h"
#include <sys/time.h>



FluidSystem::FluidSystem(std::shared_ptr<SPHSolverGPU> _fluidSolver,
                         std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    m_fluidSolver = _fluidSolver;
    m_fluid = _fluid;
    m_fluidSolverProperty = _fluidSolverProperty;
}

FluidSystem::FluidSystem(const FluidSystem &_FluidSystem)
{

}

FluidSystem::~FluidSystem()
{
    m_fluidSolver = nullptr;
    m_fluid = nullptr;
    m_fluidSolverProperty = nullptr;
}

void FluidSystem::AddFluidSolver(std::shared_ptr<SPHSolverGPU> _fluidSolver)
{
    m_fluidSolver = _fluidSolver;
}

void FluidSystem::AddFluid(std::shared_ptr<Fluid> _fluid)
{
    m_fluid = _fluid;
}

void FluidSystem::AddAlgae(std::shared_ptr<Fluid> _algae)
{
    m_algae = _algae;
}

void FluidSystem::AddFluidSolverProperty(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    m_fluidSolverProperty = _fluidSolverProperty;
}

void FluidSystem::InitialiseSim()
{
    float3 *pos = m_fluid->GetPositionPtr();
    float3 *vel = m_fluid->GetVelocityPtr();
    float *den = m_fluid->GetDensityPtr();
    float *mass = m_fluid->GetMassPtr();

    m_fluidSolver->InitFluidAsCube(pos, vel, den,
                                   m_fluid->GetFluidProperty()->restDensity,
                                   m_fluid->GetFluidProperty()->numParticles,
                                   ceil(cbrt(m_fluid->GetFluidProperty()->numParticles)),
                                   2.0f*m_fluid->GetFluidProperty()->particleRadius);

    cudaThreadSynchronize();

    m_fluid->ReleasePositionPtr();
    m_fluid->ReleaseVelocityPtr();
    m_fluid->ReleaseDensityPtr();
    m_fluid->ReleaseMassPtr();
}

void FluidSystem::ResetSim()
{
    float3 *pos = m_fluid->GetPositionPtr();
    float3 *vel = m_fluid->GetVelocityPtr();
    float *den = m_fluid->GetDensityPtr();
    float *mass = m_fluid->GetMassPtr();

    m_fluidSolver->InitFluidAsCube(pos, vel, den,
                                   m_fluid->GetFluidProperty()->restDensity,
                                   m_fluid->GetFluidProperty()->numParticles,
                                   ceil(cbrt(m_fluid->GetFluidProperty()->numParticles)),
                                   2.0f*m_fluid->GetFluidProperty()->particleRadius);

    cudaThreadSynchronize();

    m_fluid->ReleasePositionPtr();
    m_fluid->ReleaseVelocityPtr();
    m_fluid->ReleaseDensityPtr();
    m_fluid->ReleaseMassPtr();
}

void FluidSystem::StepSimulation()
{
    if(!m_fluid->GetFluidProperty()->play)
    {
        return;
    }


    static double time = 0.0;
    static double t1 = 0.0;
    static double t2 = 0.0;
    struct timeval tim;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec/1000000.0);


    // map the buffer to our CUDA device pointer
    float3 *pos = m_fluid->GetPositionPtr();
    float3 *vel = m_fluid->GetVelocityPtr();
    float *den = m_fluid->GetDensityPtr();
    float *mass = m_fluid->GetMassPtr();


    // Simulate here
    m_fluidSolver->Solve(m_fluidSolver->GetFluidSolverProperty()->deltaTime, pos, vel, den);
    cudaThreadSynchronize();


    // Clean up
    m_fluid->ReleasePositionPtr();
    m_fluid->ReleaseVelocityPtr();
    m_fluid->ReleaseDensityPtr();
    m_fluid->ReleaseMassPtr();


    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    time += 10*(t2-t1);
    //std::cout<<"dt: "<<t2-t1<<"\n";
}
