#version 330
in vec3 fPos;
in vec3 fVel;
in float fDen;

out vec4 fragColor;

uniform vec3 uLightPos;
uniform vec3 uColour;


void main()
{

    fragColor = vec4(uColour, 1.0);
}
