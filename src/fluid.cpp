#include "include/fluid.h"
#include <math.h>
#include<sys/time.h>
#include <glm/gtx/transform.hpp>

Fluid::Fluid(FluidProperty *_fluidProperty)
{
    m_fluidProperty = _fluidProperty;
    m_solver = new SPHSolverGPU(m_fluidProperty);

    m_colour = glm::vec3(0.8f, 0.4f, 0.4f);

    Init();
}

Fluid::~Fluid()
{
    delete m_fluidProperty;
    delete m_solver;

    cudaGraphicsUnregisterResource(m_posBO_CUDA);
    m_posBO.destroy();

    cudaGraphicsUnregisterResource(m_velBO_CUDA);
    m_velBO.destroy();

    cudaGraphicsUnregisterResource(m_denBO_CUDA);
    m_denBO.destroy();

    m_meshIBO.destroy();
    m_meshVBO.destroy();
    m_meshNBO.destroy();
    m_vao.destroy();
    m_shaderProg.destroyed();
}

void Fluid::Init()
{
    cudaSetDevice(0);


    float dia = 2.0f * m_fluidProperty->particleRadius;
    m_fluidProperty->particleMass = m_fluidProperty->restDensity * (dia * dia * dia);
    std::cout<<"particle mass: "<<m_fluidProperty->particleMass<<"\n";
    AppendSphereVerts(glm::vec3(0.0f,0.0f,0.0f), m_fluidProperty->particleRadius);

    InitGL();
    InitParticles();

}

void Fluid::Simulate()
{
    static double time = 0.0;
    static double t1 = 0.0;
    static double t2 = 0.0;
    struct timeval tim;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec/1000000.0);


    // map the buffer to our CUDA device pointer
    cudaGraphicsMapResources(1, &m_posBO_CUDA, 0);
    size_t numBytesPos;
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions_ptr, &numBytesPos, m_posBO_CUDA);

    cudaGraphicsMapResources(1, &m_velBO_CUDA, 0);
    size_t numBytesVel;
    cudaGraphicsResourceGetMappedPointer((void **)&d_velocities_ptr, &numBytesVel, m_velBO_CUDA);

    cudaGraphicsMapResources(1, &m_denBO_CUDA, 0);
    size_t numBytesDen;
    cudaGraphicsResourceGetMappedPointer((void **)&d_densities_ptr, &numBytesDen, m_denBO_CUDA);


    // Simulate here
    m_solver->Solve(m_fluidProperty->deltaTime, d_positions_ptr, d_velocities_ptr, d_densities_ptr);
    cudaThreadSynchronize();


    // Clean up
    cudaGraphicsUnmapResources(1, &m_posBO_CUDA, 0);
    cudaGraphicsUnmapResources(1, &m_velBO_CUDA, 0);
    cudaGraphicsUnmapResources(1, &m_denBO_CUDA, 0);


    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    time += 10*(t2-t1);
    //std::cout<<"dt: "<<t2-t1<<"\n";
}

void Fluid::Draw()
{
    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_CULL_FACE);
    glEnable(GL_FRONT_AND_BACK);

    m_shaderProg.bind();
    m_vao.bind();
    //glDrawElementsInstanced(GL_TRIANGLES, m_meshTris.size()*3, GL_UNSIGNED_INT, &m_meshTris[0], m_fluidProperty->numParticles);
    glDrawArrays(GL_POINTS, 0, m_fluidProperty->numParticles);
    m_vao.release();
    m_shaderProg.release();

}


void Fluid::SetShaderUniforms(const glm::mat4 &_projMat, const glm::mat4 &_viewMat, const glm::mat4 &_modelMat, const glm::mat4 &_normalMat, const glm::vec3 &_lightPos, const glm::vec3 &_camPos)
{
    m_shaderProg.bind();
    glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glUniform3fv(m_colourLoc, 1, &m_colour[0]);
    glUniform1f(m_radLoc, m_fluidProperty->particleRadius);

    m_shaderProg.release();

}



void Fluid::InitGL()
{
    InitShader();
    InitVAO();
}

void Fluid::InitShader()
{
    // Create shaders
//    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/instanceVert.glsl");
//    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/instanceFrag.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/sphereSpriteVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/sphereSpriteGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/sphereSpriteFrag.glsl");
    m_shaderProg.link();

    // Get shader uniform and sttribute locations
    m_shaderProg.bind();

    m_projMatrixLoc = m_shaderProg.uniformLocation("uProjMatrix");
    m_mvMatrixLoc = m_shaderProg.uniformLocation("uMVMatrix");
    m_normalMatrixLoc = m_shaderProg.uniformLocation("uNormalMatrix");
    m_lightPosLoc = m_shaderProg.uniformLocation("uLightPos");

    m_colourLoc = m_shaderProg.uniformLocation("uColour");
    m_vertexAttrLoc = m_shaderProg.attributeLocation("vertex");
    m_normalAttrLoc = m_shaderProg.attributeLocation("normal");
    m_posAttrLoc = m_shaderProg.attributeLocation("vPos");
    m_velAttrLoc = m_shaderProg.attributeLocation("vVel");
    m_denAttrLoc = m_shaderProg.attributeLocation("vDen");
    m_radLoc = m_shaderProg.uniformLocation("uRad");
    m_camPosLoc = m_shaderProg.uniformLocation("uCameraPos");

    m_shaderProg.release();

//    std::cout<<"vertex attr loc:\t"<<m_vertexAttrLoc<<"\n";
//    std::cout<<"normal attr loc:\t"<<m_normalAttrLoc<<"\n";
//    std::cout<<"pos attr loc:\t"<<m_posAttrLoc<<"\n";
//    std::cout<<"vel attr loc:\t"<<m_velAttrLoc<<"\n";
//    std::cout<<"den attr loc:\t"<<m_denAttrLoc<<"\n";

//    std::cout<<"num verts:\t"<<m_meshVerts.size()<<"\n";
//    std::cout<<"num norms:\t"<<m_meshNorms.size()<<"\n";

}

void Fluid::InitVAO()
{
    m_shaderProg.bind();

    // Set up the VAO
    m_vao.create();
    m_vao.bind();


    // Setup element array.
//    m_meshIBO.create();
//    m_meshIBO.bind();
//    m_meshIBO.allocate(&m_meshTris[0], m_meshTris.size() * sizeof(glm::ivec3));
//    m_meshIBO.release();


//    // Setup our vertex buffer object.
//    m_meshVBO.create();
//    m_meshVBO.bind();
//    m_meshVBO.allocate(&m_meshVerts[0], m_meshVerts.size() * sizeof(glm::vec3));
//    glEnableVertexAttribArray(m_vertexAttrLoc);
//    glVertexAttribPointer(m_vertexAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
//    m_meshVBO.release();


//    // set up normal buffer object
//    m_meshNBO.create();
//    m_meshNBO.bind();
//    m_meshNBO.allocate(&m_meshNorms[0], m_meshNorms.size() * sizeof(glm::vec3));
//    glEnableVertexAttribArray(m_normalAttrLoc);
//    glVertexAttribPointer(m_normalAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
//    m_meshNBO.release();



    // Setup our pos buffer object.
    m_posBO.create();
    m_posBO.bind();
    m_posBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glEnableVertexAttribArray(m_posAttrLoc);
    glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
//    glVertexAttribDivisor(m_posAttrLoc, 1);
    m_posBO.release();
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    // Set up velocity buffer object
    m_velBO.create();
    m_velBO.bind();
    m_velBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glEnableVertexAttribArray(m_velAttrLoc);
    glVertexAttribPointer(m_velAttrLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
//    glVertexAttribDivisor(m_velAttrLoc, 1);
    m_velBO.release();
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    // Set up density buffer object
    m_denBO.create();
    m_denBO.bind();
    m_denBO.allocate(m_fluidProperty->numParticles * sizeof(float));
    glEnableVertexAttribArray(m_denAttrLoc);
    glVertexAttribPointer(m_denAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
//    glVertexAttribDivisor(m_denAttrLoc, 1);
    m_denBO.release();
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    glPointSize(5);
    m_vao.release();

    m_shaderProg.release();
}

void Fluid::InitParticles()
{
    size_t numBytes;
    cudaGraphicsMapResources(1, &m_posBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions_ptr, &numBytes, m_posBO_CUDA);

    size_t numBytesVel;
    cudaGraphicsMapResources(1, &m_velBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_velocities_ptr, &numBytesVel, m_velBO_CUDA);

    size_t numBytesDen;
    cudaGraphicsMapResources(1, &m_denBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_densities_ptr, &numBytesDen, m_denBO_CUDA);

    m_solver->InitParticleAsCube(d_positions_ptr, d_velocities_ptr, d_densities_ptr, m_fluidProperty->restDensity, m_fluidProperty->numParticles, ceil(cbrt(m_fluidProperty->numParticles)), 2.0f*m_fluidProperty->particleRadius);
    cudaThreadSynchronize();

    cudaGraphicsUnmapResources(1, &m_posBO_CUDA, 0);
    cudaGraphicsUnmapResources(1, &m_velBO_CUDA, 0);
    cudaGraphicsUnmapResources(1, &m_denBO_CUDA, 0);
}


void Fluid::AppendSphereVerts(glm::vec3 _pos, float _radius, int _stacks, int _slices)
{
    /*
     * This code is based on an answer given from the site below:
     * http://gamedev.stackexchange.com/questions/16585/how-do-you-programmatically-generate-a-sphere
     *
    */


    int indexOffset = m_meshVerts.size();

    // Generate sphere verts/normals
    for( int t = 1 ; t < _stacks-1 ; t++ )
    {
        float theta1 = ( (float)(t)/(_stacks-1) )*glm::pi<float>();

        for( int p = 0 ; p < _slices ; p++ )
        {
            float phi1 = ( (float)(p)/(_slices-1) )*2*glm::pi<float>();

            glm::vec3 vert = glm::vec3(sin(theta1)*cos(phi1), cos(theta1), -sin(theta1)*sin(phi1));
            m_meshVerts.push_back(_pos + (_radius*vert));
            m_meshNorms.push_back(vert);
        }
    }
    m_meshVerts.push_back(_pos + (_radius * glm::vec3(0.0f, 1.0f, 0.0f)));
    m_meshVerts.push_back(_pos + (_radius * glm::vec3(0.0f, -1.0f, 0.0f)));

    m_meshNorms.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
    m_meshNorms.push_back(glm::vec3(0.0f, -1.0f, 0.0f));



    // Generate sphere element array
    indexOffset = indexOffset < 0 ? 0 : indexOffset;
    for( int t = 0 ; t < _stacks-3 ; t++ )
    {
        for( int p = 0 ; p < _slices-1 ; p++ )
        {
            glm::vec3 tri1(indexOffset + ((t  )*_slices + p  ),
                           indexOffset + ((t+1)*_slices + p+1),
                           indexOffset + ((t  )*_slices + p+1));
            m_meshTris.push_back(tri1);

            glm::vec3 tri2(indexOffset + ((t  )*_slices + p  ),
                           indexOffset + ((t+1)*_slices + p  ),
                           indexOffset + ((t+1)*_slices + p+1));
            m_meshTris.push_back(tri2);
        }
    }
    // element array for top and bottom row of tri's connecting to poles
    for( int p = 0 ; p < _slices-1 ; p++ )
    {
        glm::vec3 tri1(indexOffset + ((_stacks-2)*(_slices)),
                       indexOffset + (p),
                       indexOffset + (p+1));
        m_meshTris.push_back(tri1);

        glm::vec3 tri2(indexOffset + ((_stacks-2)*_slices +1),
                       indexOffset + ((_stacks-3)*_slices +1+p),
                       indexOffset + ((_stacks-3)*_slices +p));
        m_meshTris.push_back(tri2);
    }

}

