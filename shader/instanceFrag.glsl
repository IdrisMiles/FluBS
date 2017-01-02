#version 150
in vec3 vert;
in vec3 vertNormal;
in vec3 fDen;

out vec4 fragColor;

uniform vec3 uLightPos;
uniform vec3 uColour;


void main()
{
    vec3 L = normalize(uLightPos - vert);
    float NL = max(dot(normalize(vertNormal), L), 0.0);
    vec3 col = clamp((fDen * 0.2) + (fDen * 0.4) + (fDen * 0.6 * NL), 0.0, 1.0);
    fragColor = vec4(col, 1.0);
}
