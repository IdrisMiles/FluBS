#version 330
in vec3 vPos;
in vec3 vVel;
in float vDen;

out vec3 gPos;
out vec3 gVel;
out float gDen;

uniform mat4 uProjMatrix;
uniform mat4 uMVMatrix;
uniform mat3 uNormalMatrix;


void main()
{
   gPos = vPos;
   gVel = vVel;
   gDen = vDen;

   gl_Position = /*uProjMatrix * uMVMatrix * */vec4(vPos,1.0);
}
