#version 430

in vec3 vPos;
in vec2 vUV;

out vec2 fUV;


void main()
{
   fUV = vUV;

   gl_Position =  vec4(vPos,1.0);
}
