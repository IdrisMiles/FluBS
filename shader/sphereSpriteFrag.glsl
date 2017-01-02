#version 330
in vec3 fPos;
in vec3 fVel;
in float fDen;
in vec2 fTexCoord;

out vec4 fragColor;

uniform vec3 uLightPos;
uniform vec3 uColour;
uniform float uRad;


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
    else
    {
        z = sqrt(z2);
//        gl_FragDepth = gl_FragCoord.z + z;

        vec3 norm = vec3(x,y,z);

        vec3 L = normalize(uLightPos - fPos);
        float NL = max(dot(normalize(norm), L), 0.0);
        vec3 densityColour = vec3(0,0,0);

        float densityRatio = fDen/1000.0f;
        if(densityRatio < 1.0)
        {
            densityColour  = mix(vec3(1,0,0), vec3(0,1,0), densityRatio);
        }
        else
        {
            densityColour  = mix(vec3(0,1,0), vec3(1,1,1), densityRatio-1.0);
        }

        vec3 shadedColour = clamp((densityColour * 0.4) + (densityColour * 0.6 * NL), 0.0, 1.0);
//        vec3 shadedColour = clamp((uColour * 0.4) + (uColour * 0.6 * NL), 0.0, 1.0);

        fragColor = vec4(shadedColour, 1.0);
    }

}
