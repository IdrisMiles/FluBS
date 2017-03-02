#version 430

in vec3 fVel;
in float fDen;
in vec2 fTexCoord;

layout (location = 0) out vec4 oDepth;

uniform float near = 0.02f;
uniform float far = 30.0f;
uniform float uRestDen = 1000.0f;

void main()
{
    if(fDen < 0.5f*uRestDen)
    {
        discard;
        return;
    }

    float x = 2.0f * (fTexCoord.x - 0.5f);
    float y = 2.0f * (fTexCoord.y - 0.5f);
    float z2 = 1.0 - ((x*x)+(y*y));

    if(z2 < 0.0f)
    {
        discard;
        return;
    }

//    oDepth = vec4(vec3(0.1f / (far+near)), 1.0f);
    oDepth = vec4(vec3(1.0f/far+near), 1.0f);
}
