#version 430
#extension GL_NV_shadow_samplers_cube : enable

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
    vec4 depth = texture2D(uDepthTex, fUV);
    vec4 thickness = texture2D(uThicknessTex, fUV);
    if(depth.a <= 0.1f || thickness.r <= 0.0f)
    {
        discard;
    }


    //------------------------------------------------------------
    // Bioluminescent algae stuff
    vec4 algaeDepth = texture2D(uAlgaeDepthTex, fUV).rgba;
    vec4 algaeThickness = texture2D(uAlgaeThicknessTex, fUV).rgba;

    if(algaeDepth.a <= 0.1f || algaeThickness.r <= 0.0001f)// || algaeThickness.a <= 0.0001f)
    {
        algaeDepth.rgb = vec3(0.0f);
        algaeThickness.rgb = vec3(0.0f);
    }

    float biolumAtten = 1.0f - clamp(((1.0f - depth.b) - (1.0f - algaeDepth.b)), 0.0f, 1.0f);
    float biolumIntensity = algaeThickness.r;
    vec3 biolumColour = biolumAtten * biolumIntensity * vec3(1.0f, 1.0f, 1.0f);


    //------------------------------------------------------------
    // Get position
    float h = 0.005f;
    float h2 = 2.0f*h;
    float px = (fUV.x - 0.5f) * 2.0f;
    float py = (fUV.y - 0.5f) * 2.0f;
    vec3 pos = vec3(px, py, depth.b);


    //------------------------------------------------------------
    // Get normal
    float nx = (texture2D(uDepthTex, fUV + vec2(h, 0.0f)).b - texture2D(uDepthTex, fUV + vec2(-h, 0.0f)).b) / h2;
    float ny = (texture2D(uDepthTex, fUV + vec2(0.0f, h)).b - texture2D(uDepthTex, fUV + vec2(0.0f, -h)).b) / h2;
    vec3 normal = normalize(vec3(nx, ny, 1.0f));

    //------------------------------------------------------------
    // Eye ray
    vec3 eye = uCameraPos - pos;

    //------------------------------------------------------------
    // Get reflected ray and colour
    vec3 reflectRay = reflect(eye, normal);
    vec3 reflectColour = vec3(0.2f);//textureCube(uCubeMapTex, reflectRay).rgb;


    //------------------------------------------------------------
    // Get refracted ray and colour
    vec3 refractRay = refract(eye, normal, 1.3f);
    vec3 refractColour = vec3(0.2f);// textureCube(uCubeMapTex, refractRay).rgb;
    float attenuation = thickness.r;


    //------------------------------------------------------------
    // final colour
    fragColor.rgb = (reflectColour*0.4f) + ((1.0f - attenuation) * refractColour) + (uFluidColour * attenuation) + biolumColour;
//    fragColor.rgb = vec3(depth.b);
//    fragColor.rgb = thickness.rgb;
//    fragColor.rgb = algaeThickness.rgb;
//    fragColor.rgb = algaeDepth.rgb;
}
