#version 430

in vec2 fUV;

uniform vec3 uCameraPos;
uniform vec3 uFluidColour = vec3(0.1f,0.4f,0.9f);

uniform sampler2D uDepthTex;
uniform sampler2D uThicknessTex;
uniform sampler2D uAlgaeDepthTex;
uniform sampler2D uAlgaeThicknessTex;
uniform samplerCube uCubeMapTex;

uniform mat4 uInvProjMatrix;
uniform mat4 uNormalMatrix;

out vec4 fragColor;


//-----------------------------------------------------------------------------------------------------------------

vec3 UvToEye(vec2 _uv, float _depth)
{
    return vec3(0.0f, 0.0f, _depth);
}

//-----------------------------------------------------------------------------------------------------------------

void main()
{
    vec4 depth = texture(uDepthTex, fUV);
    vec4 thickness = texture(uThicknessTex, fUV);
    if(depth.a <= 0.1f || thickness.r <= 0.0f)
    {
        discard;
    }

    vec4 algaeDepth = texture(uAlgaeDepthTex, fUV);
    vec4 algaeThickness = texture(uAlgaeThicknessTex, fUV);
    if(algaeDepth.a <= 0.1f || algaeThickness.r <= 0.0001f || algaeThickness.a <= 0.0001f)
    {
        algaeDepth.rgb = vec3(0.0f);
        algaeThickness.rgb = vec3(0.0f);
    }
    else
    {
//        algaeDepth.rgb = vec3(0.0f);
//        algaeThickness.rgb = vec3(0.1f);
    }

    float biolumAtten = 1.0f - clamp(((1.0f - depth.r) - (1.0f - algaeDepth.r)), 0.0f, 1.0f);
    float biolumIntensity = algaeThickness.r;
    vec3 biolumColour = biolumAtten * biolumIntensity * vec3(1.0f, 1.0f, 1.0f);

    float h = 0.005f;
    float h2 = 2.0f*h;

    // Get position
    float px = (fUV.x - 0.5f) * 2.0f;
    float py = (fUV.y - 0.5f) * 2.0f;
    vec3 pos = vec3(px, py, depth.r);

    // Get normal
    float nx = (texture(uDepthTex, fUV + vec2(h, 0.0f)).r - texture(uDepthTex, fUV + vec2(-h, 0.0f)).r) / h2;
    float ny = (texture(uDepthTex, fUV + vec2(0.0f, h)).r - texture(uDepthTex, fUV + vec2(0.0f, -h)).r) / h2;
    vec3 normal = normalize(vec3(nx, ny, 1.0f));

    // Eye ray
    vec3 eye = uCameraPos - pos;

    // Get reflected ray and colour
    vec3 reflectRay = reflect(eye, normal);
    vec3 reflectColour = texture(uCubeMapTex, reflectRay).rgb;

    // Get refracted ray and colour
    vec3 refractRay = refract(eye, normal, 1.3f);
    vec3 refractColour = texture(uCubeMapTex, refractRay).rgb;
    float attenuation = thickness.r;


    // final colour
    fragColor.rgb = (reflectColour*0.4f) + ((1.0f - attenuation) * refractColour) + (uFluidColour * attenuation) + biolumColour;
//    fragColor.rgb = depth.rgb;
//    fragColor.rgb = thickness.rgb;
//    fragColor.rgb = algaeThickness.rgb;
//    fragColor.rgb = algaeDepth.rgb;
}
