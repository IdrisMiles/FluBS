#version 430

in vec3 fVel;
in float fDen;
in vec2 fTexCoord;
in float fBio;

layout (location = 0) out vec4 oDepth;

uniform float uRestDen = 1000.0f;

void main()
{
    float x = 2.0f * (fTexCoord.x - 0.5f);
    float y = 2.0f * (fTexCoord.y - 0.5f);
    float z2 = ((x*x)+(y*y));

    if(z2 > 1.0f)
    {
        discard;
        return;
    }

    oDepth = vec4(vec3(fBio, fBio, fBio), 1.0f);
}
