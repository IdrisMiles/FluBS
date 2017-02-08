#include "Rigid/boundary.h"

Boundary::Boundary(std::shared_ptr<FluidProperty> _fluidProperty, Mesh _mesh) : Fluid(_fluidProperty)
{
    m_colour = glm::vec3(0.6f, 0.3f, 0.1f);

    if(_mesh.verts.size() != _fluidProperty->numParticles)
    {
        // Warning shit don't match!
        // Throw toys
    }

    m_fluidProperty = _fluidProperty;
    m_mesh = _mesh;

    cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_fluidProperty->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
}


Boundary::~Boundary()
{
    m_fluidProperty = nullptr;
    CleanUpCUDAMemory();
    CleanUpGL();
}


float *Boundary::GetVolumePtr()
{
    return d_volumePtr;
}

void Boundary::ReleaseVolumePtr()
{

}
