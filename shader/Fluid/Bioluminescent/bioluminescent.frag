#version 430
#extension GL_NV_shadow_samplers_cube : enable

in vec2 fUV;

uniform vec3 uCameraPos;
uniform vec3 uFluidColour = vec3(0.1f,0.4f,0.9f);
uniform vec3 uLightColour = vec3(1.0f, 1.0f, 1.0f);
uniform vec3 uLightPos;

uniform sampler2D uDepthTex;
uniform sampler2D uThicknessTex;
uniform sampler2D uAlgaeDepthTex;
uniform sampler2D uAlgaeThicknessTex;
uniform samplerCube uCubeMapTex;

uniform mat4 uInvPVMatrix;
uniform mat3 uNormalMatrix;

out vec4 fragColor;


//-----------------------------------------------------------------------------------------------------------------

vec3 ScreenToWorld(vec2 _uv, float _depth)
{
    // ndc [1:-1]
    vec4 ndc = vec4((_uv*2.0f)-vec2(1.0f, 1.0f), (_depth*2.0f)-1.0f, 1.0f);
    vec4 pos = uInvPVMatrix * ndc;
    pos.w = 1.0f / pos.w;
    pos.xyz *= pos.w;

    return pos.xyz;
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
    vec3 biolumColour = abs(biolumAtten * biolumIntensity) * vec3(1.0f, 1.0f, 1.0f);


    //------------------------------------------------------------
    // Get position
    vec3 pos = ScreenToWorld(fUV, depth.b);


    //------------------------------------------------------------
    // Get normal
    float h = 0.005f;

    vec3 ddx = ScreenToWorld(fUV + vec2(h, 0.0f), texture2D(uDepthTex, fUV + vec2(h, 0.0f)).b) - pos;
    vec3 ddx2 = pos - ScreenToWorld(fUV + vec2(-h, 0.0f), texture2D(uDepthTex, fUV + vec2(-h, 0.0f)).b);
    if(abs(ddx.z) > abs(ddx2.z))
    {
        ddx = ddx2;
    }

    vec3 ddy = ScreenToWorld(fUV + vec2(0.0f, h), texture2D(uDepthTex, fUV + vec2(0.0f, h)).b) - pos;
    vec3 ddy2 = pos - ScreenToWorld(fUV + vec2(0.0f, -h), texture2D(uDepthTex, fUV + vec2(0.0f, -h)).b);
    if(abs(ddy.z) > abs(ddy2.z))
    {
        ddy = ddy2;
    }

    vec3 normal = uNormalMatrix * normalize(cross(ddx, ddy));


    //------------------------------------------------------------
    // Eye ray
    vec3 eye = normalize(uCameraPos - pos);
    float nDotE = dot(normal, eye);


    //------------------------------------------------------------
    // Light ray
    vec3 light = normalize(uLightPos - pos);


    //------------------------------------------------------------
    // Diffuse and Spec Shading
    vec3 diffuse = uFluidColour * dot(normal, light);


    //------------------------------------------------------------
    // Get reflected ray and colour
    vec3 reflectRay = reflect(eye, normal);
//    reflectRay = (nDotE < 0.0f) ? -reflectRay : reflectRay;
    vec3 reflectColour = textureCube(uCubeMapTex, reflectRay).rgb;


    //------------------------------------------------------------
    // Get refracted ray and colour
    vec3 refractRay = refract(eye, normal, 1.3f);
    vec3 refractColour = textureCube(uCubeMapTex, refractRay).rgb;
    float attenuation = thickness.r;


    //------------------------------------------------------------
    // final colour
    fragColor.rgb = (reflectColour*0.4f) + ((1.0f - attenuation) * refractColour) + (uFluidColour * attenuation) + biolumColour;
    fragColor.a = (attenuation);
//    fragColor.rgb = normal;
//    fragColor.rgb = diffuse;
//    fragColor.rgb = reflectColour;
//    fragColor.rgb = refractColour;
//    fragColor.rgb = vec3(depth.g);
//    fragColor.rgb = thickness.rgb;
//    fragColor.rgb = algaeThickness.rgb;
//    fragColor.rgb = algaeDepth.rgb;
    gl_FragDepth = depth.g;
}
