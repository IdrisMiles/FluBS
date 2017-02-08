#ifndef RENDERMESH_H
#define RENDERMESH_H


/// @author Idris Miles
/// @version 0.1.0
/// @date 10th January 2017


// Open GL includes
#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

#include "Mesh/mesh.h"

#include <memory>

/// @class RenderMesh class. This class render a skinned mesh, using PhysicsBody to deform the mesh.
class RenderMesh
{
public:
    RenderMesh();
    RenderMesh(const RenderMesh &_copy);
    virtual ~RenderMesh();

    virtual void LoadMesh(const Mesh &_mesh);
    virtual void DrawMesh();

    // Setter
    void SetWireframe(const bool &_wireframe);
    void SetDrawMesh(const bool &_drawMesh);
    void SetColour(const glm::vec3 &_colour);


protected:
    void CreateShader();
    void CreateVAOs();
    void DeleteVAOs();
    void UpdateVAOs();



    bool m_wireframe;
    bool m_drawMesh;
    bool m_meshLoaded;
    bool m_vaoLoaded;

    glm::mat4 m_modelMat;
    glm::vec3 m_colour;

    // Scene shader stuff
    int m_projMatrixLoc;
    int m_mvMatrixLoc;
    int m_viewMatrixLoc;
    int m_modelMatrixLoc;
    int m_normalMatrixLoc;
    int m_lightPosLoc;
    int m_colourLoc;

    QOpenGLShaderProgram *m_shaderProg;
    QOpenGLVertexArrayObject m_meshVAO;
    QOpenGLBuffer m_meshVBO;
    QOpenGLBuffer m_meshNBO;
    QOpenGLBuffer m_meshIBO;
    QOpenGLBuffer m_meshModelMatInstanceBO;

    Mesh m_mesh;

};

#endif // RENDERMESH_H
