#include "include/fluidsystem.h"
#include <sys/time.h>



FluidSystem::FluidSystem(std::shared_ptr<SPHSolverGPU> _fluidSolver,
                         std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _fluidSolverProperty,
                         std::shared_ptr<FluidProperty> _fluidProperty)
{
    m_fluidSolver = _fluidSolver;
    m_fluid = _fluid;
    m_fluidSolverProperty = _fluidSolverProperty;
    m_fluidProperty = _fluidProperty;
}

FluidSystem::FluidSystem(const FluidSystem &_FluidSystem)
{

}

FluidSystem::~FluidSystem()
{
    m_fluidSolver = nullptr;
    m_fluid = nullptr;
    m_fluidSolverProperty = nullptr;
    m_fluidProperty = nullptr;
}

void FluidSystem::AddFluidSolver(std::shared_ptr<SPHSolverGPU> _fluidSolver)
{
    m_fluidSolver = _fluidSolver;
}

void FluidSystem::AddFluid(std::shared_ptr<Fluid> _fluid)
{
    m_fluid = _fluid;
}

void FluidSystem::AddFluidSolverProperty(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    m_fluidSolverProperty = _fluidSolverProperty;
}

void FluidSystem::AddFluidProperty(std::shared_ptr<FluidProperty> _fluidProperty)
{
    m_fluidProperty = _fluidProperty;
}

void FluidSystem::InitialiseSim()
{
    float3 *pos = m_fluid->GetPositionsPtr();
    float3 *vel = m_fluid->GetVelocitiesPtr();
    float *den = m_fluid->GetDensitiesPtr();

    m_fluidSolver->InitFluidAsCube(pos, vel, den,
                                   m_fluidProperty->restDensity,
                                   m_fluidProperty->numParticles,
                                   ceil(cbrt(m_fluidProperty->numParticles)),
                                   2.0f*m_fluidProperty->particleRadius);

    cudaThreadSynchronize();

    m_fluid->ReleasePositionsPtr();
    m_fluid->ReleaseVelocitiesPtr();
    m_fluid->ReleaseDensitiesPtr();
}

void FluidSystem::ResetSim()
{
    float3 *pos = m_fluid->GetPositionsPtr();
    float3 *vel = m_fluid->GetVelocitiesPtr();
    float *den = m_fluid->GetDensitiesPtr();

    m_fluidSolver->InitFluidAsCube(pos, vel, den,
                                   m_fluidProperty->restDensity,
                                   m_fluidProperty->numParticles,
                                   ceil(cbrt(m_fluidProperty->numParticles)),
                                   2.0f*m_fluidProperty->particleRadius);

    cudaThreadSynchronize();

    m_fluid->ReleasePositionsPtr();
    m_fluid->ReleaseVelocitiesPtr();
    m_fluid->ReleaseDensitiesPtr();
}

void FluidSystem::StepSimulation()
{
    if(!m_fluidProperty->play)
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
    float3 *pos = m_fluid->GetPositionsPtr();
    float3 *vel = m_fluid->GetVelocitiesPtr();
    float *den = m_fluid->GetDensitiesPtr();


    // Simulate here
    m_fluidSolver->Solve(m_fluidProperty->deltaTime, pos, vel, den);
    cudaThreadSynchronize();


    // Clean up
    m_fluid->ReleasePositionsPtr();
    m_fluid->ReleaseVelocitiesPtr();
    m_fluid->ReleaseDensitiesPtr();


    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    time += 10*(t2-t1);
    //std::cout<<"dt: "<<t2-t1<<"\n";
}
