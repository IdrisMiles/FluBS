#include "FluidSystem/fluidsystem.h"
#include "SPH/sph.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


FluidSystem::FluidSystem(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    if(_fluidSolverProperty != nullptr)
    {
        m_fluidSolverProperty = _fluidSolverProperty;
    }
    else
    {
        m_fluidSolverProperty.reset(new FluidSolverProperty());
    }
}

FluidSystem::FluidSystem(const FluidSystem &_FluidSystem)
{
    m_fluid = nullptr;
    m_algae = nullptr;
    m_staticRigids.clear();
    m_activeRigids.clear();
}

FluidSystem::~FluidSystem()
{
    m_fluid = nullptr;
    m_fluidSolverProperty = nullptr;
    m_staticRigids.clear();
    m_activeRigids.clear();
}

void FluidSystem::SetContainer(std::shared_ptr<Rigid> _container)
{
    cudaThreadSynchronize();
    m_container = _container;

    m_container->SetupSolveSpecs(m_fluidSolverProperty);


    sph::ResetProperties(m_container, m_fluidSolverProperty);
    cudaThreadSynchronize();

    sph::ComputeHash(m_container, m_fluidSolverProperty);
    cudaThreadSynchronize();

    sph::SortParticlesByHash(m_container);

    sph::ComputeParticleScatterIds(m_container, m_fluidSolverProperty);

    uint maxCellOcc;
    sph::ComputeMaxCellOccupancy(m_container, m_fluidSolverProperty, maxCellOcc);
    cudaThreadSynchronize();

    sph::ComputeParticleVolume(m_container, m_fluidSolverProperty);
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

        // If this is a static rigid we only have to compute all this once
        sph::ResetProperties(m_activeRigids.back(), m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Get particle hash IDs
        sph::ComputeHash(m_activeRigids.back(), m_fluidSolverProperty);
        cudaThreadSynchronize();

        // Sort particles
        sph::SortParticlesByHash(m_activeRigids.back());


        // Get Cell particle indexes - scatter addresses
        sph::ComputeParticleScatterIds(m_activeRigids.back(), m_fluidSolverProperty);


        // Find max cell occupancy
        uint maxCellOcc;
        sph::ComputeMaxCellOccupancy(m_activeRigids.back(), m_fluidSolverProperty, maxCellOcc);
        cudaThreadSynchronize();

        // compute rigid particle volumes
        sph::ComputeParticleVolume(m_activeRigids.back(), m_fluidSolverProperty);
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
    ResetFluid(m_fluid);
    ResetFluid(m_algae);
    ResetRigid(m_container);
    for(auto &&r : m_staticRigids)
    {
        ResetRigid(r);
    }
    for(auto &&r : m_activeRigids)
    {
        ResetRigid(r);
    }
}

void FluidSystem::ResetSim()
{
    ResetFluid(m_fluid);
    ResetFluid(m_algae);
    ResetRigid(m_container);
    for(auto &&r : m_staticRigids)
    {
        ResetRigid(r);
    }
    for(auto &&r : m_activeRigids)
    {
        ResetRigid(r);
    }
}

void FluidSystem::ResetRigid(std::shared_ptr<Rigid> _rigid)
{
    // If this is a static rigid we only have to compute all this once
    sph::ResetProperties(_rigid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // Get particle hash IDs
    sph::ComputeHash(_rigid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    // Sort particles
    sph::SortParticlesByHash(_rigid);

    // Get Cell particle indexes - scatter addresses
    sph::ComputeParticleScatterIds(_rigid, m_fluidSolverProperty);

    // Find max cell occupancy
    uint maxCellOcc;
    sph::ComputeMaxCellOccupancy(_rigid, m_fluidSolverProperty, maxCellOcc);
    cudaThreadSynchronize();

    // compute rigid particle volumes
    sph::ComputeParticleVolume(_rigid, m_fluidSolverProperty);


    _rigid->ReleaseCudaGLResources();
}

void FluidSystem::ResetFluid(std::shared_ptr<Fluid> _fluid)
{
    sph::ResetProperties(_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    sph::InitFluidAsCube(_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();
    _fluid->ReleaseCudaGLResources();
}

void FluidSystem::GenerateDefaultContainer()
{
    auto rigidProps = std::shared_ptr<RigidProperty>(new RigidProperty());

    Mesh boundary = Mesh();
    float dim = 0.95f* m_fluidSolverProperty->gridResolution*m_fluidSolverProperty->gridCellWidth;
    float rad = rigidProps->particleRadius;
    int numRigidAxis = ceil(dim / (rad*2.0f));
    for(int z=0; z<numRigidAxis; z++)
    {
        for(int y=0; y<numRigidAxis; y++)
        {
            for(int x=0; x<numRigidAxis; x++)
            {
                if(x==0 || x==numRigidAxis-1 || y==0 || z==0 || z==numRigidAxis-1)
                {
                    glm::vec3 pos((x*rad*2.0f)-(dim*0.5f), (y*rad*2.0f)-(dim*0.5f), (z*rad*2.0f)-(dim*0.5f));
                    boundary.verts.push_back(pos);
                }
            }
        }
    }

    rigidProps->numParticles = boundary.verts.size();
    auto container = std::shared_ptr<Rigid>(new Rigid(rigidProps, boundary));
    SetContainer(container);
}

void FluidSystem::StepSimulation()
{
    if(!(m_fluid->GetProperty())->play)
    {
        return;
    }

    if(m_container == nullptr)
    {
        GenerateDefaultContainer();
    }


    //----------------------------------------------------------------------
    // Call sph API to do funky stuff here

    sph::ResetProperties(m_fluid, m_fluidSolverProperty);
    sph::ResetProperties(m_algae, m_fluidSolverProperty);
    sph::ResetProperties(m_activeRigids, m_fluidSolverProperty);
    cudaThreadSynchronize();


    //----------------------------------------------------------------------

    sph::ComputeHash(m_fluid, m_fluidSolverProperty);
    sph::ComputeHash(m_algae, m_fluidSolverProperty);
    sph::ComputeHash(m_activeRigids, m_fluidSolverProperty);
    cudaThreadSynchronize();

    sph::SortParticlesByHash(m_fluid);
    sph::SortParticlesByHash(m_algae);
    sph::SortParticlesByHash(m_activeRigids);

    sph::ComputeParticleScatterIds(m_fluid, m_fluidSolverProperty);
    sph::ComputeParticleScatterIds(m_algae, m_fluidSolverProperty);
    sph::ComputeParticleScatterIds(m_activeRigids, m_fluidSolverProperty);

    uint maxCellOcc;
    sph::ComputeMaxCellOccupancy(m_fluid, m_fluidSolverProperty, maxCellOcc);
    sph::ComputeMaxCellOccupancy(m_algae, m_fluidSolverProperty, maxCellOcc);
    sph::ComputeMaxCellOccupancy(m_activeRigids, m_fluidSolverProperty, maxCellOcc);
    cudaThreadSynchronize();


    //----------------------------------------------------------------------

    sph::ComputeParticleVolume(m_activeRigids, m_fluidSolverProperty);
    sph::ComputeDensity(m_fluid, m_fluidSolverProperty, true, m_container, m_staticRigids, m_activeRigids, m_algae);
    sph::ComputeDensity(m_algae, m_fluidSolverProperty, true, m_container, m_staticRigids, m_activeRigids, m_fluid);
    cudaThreadSynchronize();

    sph::ComputePressure(m_fluid, m_fluidSolverProperty);
    sph::ComputePressure(m_algae, m_fluidSolverProperty);
    cudaThreadSynchronize();


    //----------------------------------------------------------------------

    sph::ComputePressureForce(m_fluid, m_fluidSolverProperty, true, m_container, m_staticRigids, m_activeRigids);
    sph::ComputeViscForce(m_fluid, m_fluidSolverProperty);
    sph::ComputeSurfaceTensionForce(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    sph::ComputeTotalForce(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();

    sph::Integrate(m_fluid, m_fluidSolverProperty);
    cudaThreadSynchronize();



    // Handle boundaries - In theory don't need to do this anymore
//    sph::HandleBoundaries(m_fluid, m_fluidSolverProperty);
//    cudaThreadSynchronize();








    //----------------------------------------------------------------------

//    sph::ResetProperties(m_fluid, m_fluidSolverProperty);
//    cudaThreadSynchronize();

//    sph::ComputeHash(m_fluid, m_fluidSolverProperty);
//    cudaThreadSynchronize();

//    sph::SortParticlesByHash(m_fluid);

//    sph::ComputeParticleScatterIds(m_fluid, m_fluidSolverProperty);

//    sph::ComputeMaxCellOccupancy(m_fluid, m_fluidSolverProperty, maxCellOcc);
//    cudaThreadSynchronize();

//    for(int i=0;i<3;i++)
//    {
//    sph::ResetTotalForce(m_fluid, m_fluidSolverProperty);

//    sph::ComputeDensity(m_fluid, m_fluidSolverProperty);
//    sph::ComputeDensity(m_fluid, m_container, m_fluidSolverProperty, true);
//    for(auto &&r : m_staticRigids)
//    {
//        sph::ComputeDensity(m_fluid, r, m_fluidSolverProperty, true);
//    }
//    cudaThreadSynchronize();

//    sph::ComputePressure(m_fluid, m_fluidSolverProperty);
//    cudaThreadSynchronize();

//    sph::ComputePressureForce(m_fluid, m_fluidSolverProperty);
//    sph::ComputePressureForce(m_fluid, m_container, m_fluidSolverProperty, true);
//    for(auto &&r : m_staticRigids)
//    {
//        sph::ComputePressureForce(m_fluid, r, m_fluidSolverProperty, true);
//    }
//    cudaThreadSynchronize();

//    sph::ComputeTotalForce(m_fluid, m_fluidSolverProperty, true, false, false, false, false);
//    cudaThreadSynchronize();

//    sph::Integrate(m_fluid, m_fluidSolverProperty);
//    cudaThreadSynchronize();
//    }

    //----------------------------------------------------------------------

    // Clean up
    m_fluid->ReleaseCudaGLResources();
    m_algae->ReleaseCudaGLResources();
    m_container->ReleaseCudaGLResources();
    for(auto &&r : m_staticRigids)
    {
        r->ReleaseCudaGLResources();
    }
    for(auto &&r : m_activeRigids)
    {
        r->ReleaseCudaGLResources();
    }
}
