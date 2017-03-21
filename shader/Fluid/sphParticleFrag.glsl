#version 430

in vec3 fPos;
in vec3 fVel;
in float fDen;
in vec2 fTexCoord;

layout (location = 0) out vec4 fragColor;

uniform vec3 uLightPos = vec3(0.0f, 0.0f, 1.0f);
uniform vec3 uColour = vec3(0.0f, 0.0f, 0.0f);
uniform float uRad = 1.0f;
uniform float uRestDen = 1000.0f;
uniform bool shadeDensity = true;


void main()
{
    float x = 2.0f * (fTexCoord.x - 0.5f);
    float y = 2.0f * (fTexCoord.y - 0.5f);
    float z2 = 1.0 - ((x*x)+(y*y));
    float z = 0.0f;

    if(z2 < 0.0f)
    {
        discard;
    }

    z = sqrt(z2);

    vec3 norm = vec3(x,y,z);

    vec3 L = normalize(uLightPos - fPos);
    float NL = max(dot(normalize(norm), L), 0.0);

    float densityRatio = fDen/uRestDen;
    vec3 densityColour = (densityRatio < 1.0) ?
                mix(vec3(1.0f,0.0f,0.0f), uColour, densityRatio) :
                mix(uColour, vec3(1.0f,1.0f,1.0f), densityRatio-1.0f);

    vec3 colour = shadeDensity ? densityColour : uColour;

    vec3 shadedColour = clamp((colour * 0.4f) + (colour * 0.6f * NL), 0.0f, 1.0f);

    fragColor = vec4(shadedColour, 1.0f);
}
