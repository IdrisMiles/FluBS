#version 430

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vVel;
layout(location = 2) in float vDen;

out vec3 gVel;
out float gDen;
out vec3 gPos;


void main()
{
    gVel = vVel;
    gDen = vDen;
    gPos = vPos;
   gl_Position = vec4(vPos,1.0);
}
