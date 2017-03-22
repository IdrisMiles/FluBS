#version 430

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vVel;
layout(location = 2) in float vDen;
layout(location = 3) in float vBio;

out vec3 gVel;
out float gDen;
out float gBio;


void main()
{
    gVel = vVel;
    gDen = vDen;
    gBio = vBio;
    gl_Position = vec4(vPos,1.0);
}
