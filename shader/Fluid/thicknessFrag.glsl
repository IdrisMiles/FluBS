#version 430

in vec3 fVel;
in float fDen;
in vec2 fTexCoord;

layout (location = 0) out vec4 oDepth;

uniform float particleThickness = 0.01f;
uniform float uRestDen = 1000.0f;

void main()
{
    if(fDen < 0.2f*uRestDen)
    {
//        discard;
//        return;
    }

    float x = 2.0f * (fTexCoord.x - 0.5f);
    float y = 2.0f * (fTexCoord.y - 0.5f);
    float z2 = 1.0 - ((x*x)+(y*y));

    if(z2 < 0.0f)
    {
        discard;
        return;
    }

    oDepth = vec4(vec3(particleThickness, particleThickness, particleThickness), 1.0f);
}
