#include "FluidSystem/fluidsystem.h"
#include <sys/time.h>
#include "SPH/sph.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


FluidSystem::FluidSystem(std::shared_ptr<Fluid> _fluid,
                         std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    m_fluid = _fluid;
    m_fluidSolverProperty = _fluidSolverProperty;
    m_fluid->SetupSolveSpecs(m_fluidSolverProperty);
}

FluidSystem::FluidSystem(const FluidSystem &_FluidSystem)
{
    m_fluid = nullptr;
    m_algae = nullptr;
    m_rigids.clear();
}

FluidSystem::~FluidSystem()
{
    m_fluid = nullptr;
    m_fluidSolverProperty = nullptr;
    m_rigids.clear();
}

void FluidSystem::AddFluid(std::shared_ptr<Fluid> _fluid)
{
    m_fluid = _fluid;
    m_fluid->SetupSolveSpecs(m_fluidSolverProperty);
}

void FluidSystem::AddRigid(std::shared_ptr<Rigid> _rigid)
{
    m_rigids.push_back(_rigid);
    m_rigids.back()->SetupSolveSpecs(m_fluidSolverProperty);
}

void FluidSystem::AddAlgae(std::shared_ptr<Fluid> _algae)
{
    m_algae = _algae;
    m_algae->SetupSolveSpecs(m_fluidSolverProperty);
}

void FluidSystem::AddFluidSolverProperty(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    m_fluidSolverProperty = _fluidSolverProperty;
}

void FluidSystem::InitialiseSim()
{
    sph::ResetProperties(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    sph::InitFluidAsCube(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    m_fluid->ReleaseCudaGLResources();


    sph::ResetProperties(m_rigids[0], m_fluidSolverProperty);
    cudaThreadSynchronize();
    m_rigids[0]->ReleaseCudaGLResources();
}

void FluidSystem::ResetSim()
{
    sph::InitFluidAsCube(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    m_fluid->ReleaseCudaGLResources();
}

void FluidSystem::StepSimulation()
{
    if(!(m_fluid->GetProperty())->play)
    {
        return;
    }


    //----------------------------------------------------------------------
    // Initialise timing stuff
    static double time = 0.0;
    static double t1 = 0.0;
    static double t2 = 0.0;
    struct timeval tim;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec/1000000.0);


    //----------------------------------------------------------------------
    // Call sph API to do funky stuff here

    for(auto &&r : m_rigids)
    {
        sph::ResetProperties(r, m_fluidSolverProperty);
    }
    sph::ResetProperties(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // Get particle hash IDs
    for(auto &&r : m_rigids)
    {
        sph::ComputeHash(r, m_fluidSolverProperty);
    }
    sph::ComputeHash(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // Sort particles
    for(auto &&r : m_rigids)
    {
        sph::SortParticlesByHash(r);
    }
    sph::SortParticlesByHash(m_fluid);

    // Get Cell particle indexes - scatter addresses
    for(auto &&r : m_rigids)
    {
        sph::ComputeParticleScatterIds(r, m_fluidSolverProperty);
    }
    sph::ComputeParticleScatterIds(m_fluid, m_fluidSolverProperty);

    uint maxCellOcc;
    for(auto &&r : m_rigids)
    {
        sph::ComputeMaxCellOccupancy(r, m_fluidSolverProperty, maxCellOcc);
    }
    sph::ComputeMaxCellOccupancy(m_fluid, m_fluidSolverProperty, maxCellOcc);
    cudaThreadSynchronize();

    if(maxCellOcc > 1024u){std::cout<<"Too many neighs\n";}

    for(auto &&r : m_rigids)
    {
        sph::ComputeParticleVolume(r, m_fluidSolverProperty);
    }

    // Compute density
    sph::ComputeDensityFluid(m_fluid, m_fluidSolverProperty, true);
    for(auto &&r : m_rigids)
    {
        sph::ComputeDensityFluidRigid(m_fluid, r, m_fluidSolverProperty, true);
        cudaThreadSynchronize();
    }

    // Compute Pressure
    sph::ComputePressureFluid(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    sph::ComputePressureForceFluid(m_fluid, m_fluidSolverProperty, true);
    sph::ComputeViscForce(m_fluid, m_fluidSolverProperty);
//    sph::ComputeSurfaceTensionForce(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    for(auto &&r : m_rigids)
    {
        sph::ComputePressureForceFluidRigid(m_fluid, r, m_fluidSolverProperty, true);
        cudaThreadSynchronize();
    }

    // Compute total force and acceleration
    sph::ComputeTotalForce(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // integrate particle position and velocities
    sph::Integrate(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // Handle boundaries
    sph::HandleBoundaries(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    //----------------------------------------------------------------------
    // Clean up
    m_fluid->ReleaseCudaGLResources();
    m_rigids[0]->ReleaseCudaGLResources();


    //----------------------------------------------------------------------
    // Get timings
    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    time += 10*(t2-t1);
    //std::cout<<"dt: "<<t2-t1<<"\n";
}
