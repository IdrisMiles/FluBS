#version 430

in vec2 fTexCoord;

layout (location = 0) out vec4 oDepth;

uniform float near = 0.1f;
uniform float far = 30.0f;


void main()
{
    oDepth = vec4(vec3(1.0f / (far+near)), 1.0f);
}
