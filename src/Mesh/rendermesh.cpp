#include "Mesh/rendermesh.h"
#include <iostream>
#include <algorithm>

#include <glm/gtx/transform.hpp>

#include <QOpenGLContext>
#include <QOpenGLFunctions>

RenderMesh::RenderMesh()
{
    m_meshLoaded = false;
    m_vaoLoaded = false;
}


RenderMesh::RenderMesh(const RenderMesh &_copy)
{
    m_meshLoaded = false;
    m_vaoLoaded = false;
}

RenderMesh::~RenderMesh()
{
    m_meshVAO.destroy();
    m_meshVBO.destroy();
    m_meshNBO.destroy();
    m_meshIBO.destroy();
    m_meshModelMatInstanceBO.destroy();

    delete m_shaderProg;
    m_shaderProg = 0;
}


void RenderMesh::LoadMesh(const Mesh &_mesh)
{
    m_meshLoaded = true;

    m_modelMat = glm::mat4(1.0f);


    m_wireframe = false;
    m_drawMesh = true;
    m_colour = glm::vec3(0.6f,0.6f,0.6f);


    //----------------------------------------------------------------------

    m_mesh.verts =_mesh.verts;
    m_mesh.norms =_mesh.norms;
    m_mesh.tris =_mesh.tris;

    //----------------------------------------------------------------------
    // Iitialise GL VAO and buffers
    if(!m_vaoLoaded)
    {
        CreateShader();
        CreateVAOs();
        UpdateVAOs();
    }
    else
    {
        UpdateVAOs();
    }

}

void RenderMesh::Draw()
{
    if(!m_drawMesh || !m_meshLoaded){return;}

    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_shaderProg->bind();

    m_meshVAO.bind();
    glPolygonMode(GL_FRONT_AND_BACK, m_wireframe?GL_LINE:GL_FILL);
    glDrawElements(GL_TRIANGLES, m_mesh.tris.size()*3, GL_UNSIGNED_INT, &m_mesh.tris[0]);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    m_meshVAO.release();

    m_shaderProg->release();
}

void RenderMesh::SetShaderUniforms(const glm::mat4 &_projMat,
                              const glm::mat4 &_viewMat,
                              const glm::mat4 &_modelMat,
                              const glm::mat4 &_normalMat,
                              const glm::vec3 &_lightPos,
                              const glm::vec3 &_camPos)
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_shaderProg->bind();

    static float i=0;
    m_modelMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    m_modelMat = glm::rotate(m_modelMat, i+=0.001f, glm::vec3(1.0f, 0.0f, 0.0f));
    glFuncs->glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*m_modelMat*_viewMat)[0][0]);
    glFuncs->glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glFuncs->glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glFuncs->glUniform3fv(m_colourLoc, 1, &m_colour[0]);
    m_shaderProg->release();

}


void RenderMesh::SetWireframe(const bool &_wireframe)
{
    m_wireframe = _wireframe;
}

void RenderMesh::SetDrawMesh(const bool &_drawMesh)
{
    m_drawMesh = _drawMesh;
}

void RenderMesh::SetColour(const glm::vec3 &_colour)
{
    m_colour = _colour;
}

//-----------------------------------------------------------------------------------------------------------------------

void RenderMesh::CreateShader()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    // setup shaders
    m_shaderProg = new QOpenGLShaderProgram;
    m_shaderProg->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/vert.glsl");
    m_shaderProg->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/frag.glsl");
    m_shaderProg->bindAttributeLocation("vertex", 0);
    m_shaderProg->bindAttributeLocation("normal", 1);
    m_shaderProg->link();

    m_shaderProg->bind();
    m_projMatrixLoc = m_shaderProg->uniformLocation("projMatrix");
    m_mvMatrixLoc = m_shaderProg->uniformLocation("mvMatrix");
    m_modelMatrixLoc = m_shaderProg->uniformLocation("modelMatrix");
    m_viewMatrixLoc = m_shaderProg->uniformLocation("viewMatrix");
    m_normalMatrixLoc = m_shaderProg->uniformLocation("normalMatrix");
    m_lightPosLoc = m_shaderProg->uniformLocation("lightPos");
    m_colourLoc = m_shaderProg->uniformLocation("colour");
}

void RenderMesh::CreateVAOs()
{
    if(m_shaderProg->bind())
    {
        QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

        glFuncs->glUniform3fv(m_colourLoc, 1, &m_colour[0]);

        m_meshVAO.create();
        m_meshVAO.bind();

        m_meshIBO.create();
        m_meshIBO.bind();
        m_meshIBO.release();


        // Setup our vertex buffer object.
        m_meshVBO.create();
        m_meshVBO.bind();
        glFuncs->glEnableVertexAttribArray( 0);
        glFuncs->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO.release();


        // Setup our normals buffer object.
        m_meshNBO.create();
        m_meshNBO.bind();
        glFuncs->glEnableVertexAttribArray(1);
        glFuncs->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO.release();


        m_meshVAO.release();

        m_shaderProg->release();

        m_vaoLoaded = true;
    }
}

void RenderMesh::DeleteVAOs()
{
    m_meshVAO.destroy();
    m_meshVBO.destroy();
    m_meshNBO.destroy();
    m_meshIBO.destroy();

    m_vaoLoaded = false;
}

void RenderMesh::UpdateVAOs()
{
    if(m_shaderProg->bind())
    {
        QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

        m_modelMatrixLoc = m_shaderProg->attributeLocation("modelMatrix");
        m_colourLoc = m_shaderProg->uniformLocation("colour");
        glFuncs->glUniform3fv(m_colourLoc, 1, &m_colour[0]);

        m_meshVAO.bind();

        m_meshIBO.bind();
        m_meshIBO.allocate(&m_mesh.tris[0], m_mesh.tris.size() * sizeof(glm::ivec3));
        m_meshIBO.release();

        // Setup our vertex buffer object.
        m_meshVBO.bind();
        m_meshVBO.allocate(&m_mesh.verts[0], m_mesh.verts.size() * sizeof(glm::vec3));
        glFuncs->glEnableVertexAttribArray( 0);
        glFuncs->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO.release();


        // Setup our normals buffer object.
        m_meshNBO.bind();
        m_meshNBO.allocate(&m_mesh.norms[0], m_mesh.norms.size() * sizeof(glm::vec3));
        glFuncs->glEnableVertexAttribArray(1);
        glFuncs->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO.release();


        m_meshVAO.release();

        m_shaderProg->release();
    }
}
