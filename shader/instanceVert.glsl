#version 150
in vec3 vertex;
in vec3 normal;
in vec3 pos; // instance data
in vec3 vel;
in float den;

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
   float d = den/1000.0;
   if(d<1.0)
   {
       vVel = mix(vec3(1,0,0), vec3(0,1,0), d);// vec3(0.0, den, 0.0);//vel;
   }
   else
   {
       vVel = mix(vec3(0,1,0), vec3(1,1,1), d-1.0);
   }
   gl_Position = projMatrix * mvMatrix * vec4(vertex + pos,1.0);
}
