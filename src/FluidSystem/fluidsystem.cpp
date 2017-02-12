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
    m_staticRigids.clear();
}

FluidSystem::~FluidSystem()
{
    m_fluid = nullptr;
    m_fluidSolverProperty = nullptr;
    m_staticRigids.clear();
}

void FluidSystem::AddFluid(std::shared_ptr<Fluid> _fluid)
{
    m_fluid = _fluid;
    m_fluid->SetupSolveSpecs(m_fluidSolverProperty);
}

void FluidSystem::AddRigid(std::shared_ptr<Rigid> _rigid)
{
    if(_rigid->GetProperty()->m_static)
    {
        m_staticRigids.push_back(_rigid);
        m_staticRigids.back()->SetupSolveSpecs(m_fluidSolverProperty);


        // If this is a static rigid we only have to compute all this once
        sph::ResetProperties(m_staticRigids.back(), m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Get particle hash IDs
        sph::ComputeHash(m_staticRigids.back(), m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Sort particles
        sph::SortParticlesByHash(m_staticRigids.back());


        // Get Cell particle indexes - scatter addresses
        sph::ComputeParticleScatterIds(m_staticRigids.back(), m_fluidSolverProperty);


        // Find max cell occupancy
        uint maxCellOcc;
        sph::ComputeMaxCellOccupancy(m_staticRigids.back(), m_fluidSolverProperty, maxCellOcc);
        cudaThreadSynchronize();

        // compute rigid particle volumes
        sph::ComputeParticleVolume(m_staticRigids.back(), m_fluidSolverProperty);
    }
    else
    {
        m_activeRigids.push_back(_rigid);
        m_activeRigids.back()->SetupSolveSpecs(m_fluidSolverProperty);
    }

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


    for(auto &&r : m_staticRigids)
    {
        // If this is a static rigid we only have to compute all this once
        sph::ResetProperties(r, m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Get particle hash IDs
        sph::ComputeHash(r, m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Sort particles
        sph::SortParticlesByHash(r);


        // Get Cell particle indexes - scatter addresses
        sph::ComputeParticleScatterIds(r, m_fluidSolverProperty);


        // Find max cell occupancy
        uint maxCellOcc;
        sph::ComputeMaxCellOccupancy(r, m_fluidSolverProperty, maxCellOcc);
        cudaThreadSynchronize();

        // compute rigid particle volumes
        sph::ComputeParticleVolume(r, m_fluidSolverProperty);


        r->ReleaseCudaGLResources();
    }
}

void FluidSystem::ResetSim()
{
    sph::InitFluidAsCube(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    m_fluid->ReleaseCudaGLResources();

    for(auto &&r : m_staticRigids)
    {
        // If this is a static rigid we only have to compute all this once
        sph::ResetProperties(r, m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Get particle hash IDs
        sph::ComputeHash(r, m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Sort particles
        sph::SortParticlesByHash(r);


        // Get Cell particle indexes - scatter addresses
        sph::ComputeParticleScatterIds(r, m_fluidSolverProperty);


        // Find max cell occupancy
        uint maxCellOcc;
        sph::ComputeMaxCellOccupancy(r, m_fluidSolverProperty, maxCellOcc);
        cudaThreadSynchronize();

        // compute rigid particle volumes
        sph::ComputeParticleVolume(r, m_fluidSolverProperty);


        r->ReleaseCudaGLResources();
    }
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

    // Reset all our sph particles here
    sph::ResetProperties(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();


    // Get particle hash IDs
    sph::ComputeHash(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // Sort particles
    sph::SortParticlesByHash(m_fluid);

    // Get Cell particle indexes - scatter addresses
    sph::ComputeParticleScatterIds(m_fluid, m_fluidSolverProperty);

    // Find max cell occupancy
    uint maxCellOcc;
    sph::ComputeMaxCellOccupancy(m_fluid, m_fluidSolverProperty, maxCellOcc);
    cudaThreadSynchronize();



    // Compute density
    sph::ComputeDensityFluid(m_fluid, m_fluidSolverProperty, true);
    for(auto &&r : m_staticRigids)
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
    for(auto &&r : m_staticRigids)
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

    // Handle boundaries - In theory don't need to do this anymore
    //sph::HandleBoundaries(m_fluid, m_fluidSolverProperty);
    //cudaThreadSynchronize();

    //----------------------------------------------------------------------
    // Clean up
    m_fluid->ReleaseCudaGLResources();
    for(auto &&r : m_staticRigids)
    {
        r->ReleaseCudaGLResources();
    }


    //----------------------------------------------------------------------
    // Get timings
    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    time += 10*(t2-t1);
    //std::cout<<"dt: "<<t2-t1<<"\n";
}
