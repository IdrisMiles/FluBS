#version 330
layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vUV;

out vec3 fPos;
out vec3 fNorm;
out vec2 fUV;


void main()
{
   fPos = vPos;
   fNorm = vNorm;
   fUV = vUV;

   gl_Position =  vec4(vPos,1.0);
}
