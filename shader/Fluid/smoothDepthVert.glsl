#version 430

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vUV;

out vec2 fUV;


void main()
{
   fUV = vUV;

   gl_Position =  vec4(vPos,1.0);
}
