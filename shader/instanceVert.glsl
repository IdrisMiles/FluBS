#version 150
in vec3 vertex;
in vec3 normal;
in vec3 pos; // instance data
in vec3 vel;

out vec3 vert;
out vec3 vertNormal;
out vec3 vVel;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat3 normalMatrix;


void main()
{
   vert = vertex.xyz;
   vertNormal = normal;
   vVel = vel;
   gl_Position = projMatrix * mvMatrix * vec4(vertex + pos,1.0);
}
