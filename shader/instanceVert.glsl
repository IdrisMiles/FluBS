#version 150
in vec3 vertex;
in vec3 normal;
in vec3 vPos; // instance data
in vec3 vVel;
in float vDen;

out vec3 vert;
out vec3 vertNormal;
out vec3 fDen;

uniform mat4 uProjMatrix;
uniform mat4 uMVMatrix;
uniform mat3 uNormalMatrix;


void main()
{
   vert = vertex.xyz;
   vertNormal = normal;
   float d = vDen/1000.0;
   if(d<1.0)
   {
       fDen = mix(vec3(1,0,0), vec3(0,1,0), d);
   }
   else
   {
       fDen = mix(vec3(0,1,0), vec3(1,1,1), d-1.0);
   }
   gl_Position = uProjMatrix * uMVMatrix * vec4(vertex + vPos,1.0);
}
