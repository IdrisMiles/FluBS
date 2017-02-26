#version 430 core

in vec3 fTexCoords;
out vec4 fragColor;

uniform samplerCube uSkyboxTex;

void main()
{
    fragColor = texture(uSkyboxTex, fTexCoords);
}
