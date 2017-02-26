#version 430 core

layout (location = 0) in vec3 vPos;

uniform mat4 uProjMatrix;
uniform mat4 uViewMatrix;

out vec3 fTexCoords;

void main()
{
    gl_Position =   uProjMatrix * uViewMatrix * vec4(vPos, 1.0);
    fTexCoords = -vPos;
}
