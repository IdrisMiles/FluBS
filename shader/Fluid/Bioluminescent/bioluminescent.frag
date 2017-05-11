#version 430
#extension GL_NV_shadow_samplers_cube : enable

in vec2 fUV;

uniform vec3 uCameraPos;
uniform vec3 uBioColour = vec3(0.6f, 0.9f, 0.3f);
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

float Fresnel(in vec3 I, in vec3 N, in float ior)
{
    float kr;
    float cosi = clamp(dot(I, N), -1.0f, 1.0f);
    float etai = 1, etat = ior;
    if (cosi > 0) { float tmp = etat; etat = etai; etai = tmp;}
    // Compute sini using Snell's law
    float sint = etai / etat * sqrt(max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1)
    {
        kr = 1;
    }
    else
    {
        float cost = sqrt(max(0.f, 1 - sint * sint));
        cosi = abs(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }

    return kr;
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
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

    if(algaeDepth.a <= 0.1f || algaeThickness.r <= 0.0001f || algaeThickness.a <= 0.0001f)
    {
        algaeDepth.rgb = vec3(0.0f);
        algaeThickness.rgb = vec3(0.0f);
    }

    float biolumAtten = 1.0f - clamp(((1.0f - depth.r) - (1.0f - algaeDepth.r)), 0.0f, 1.0f);
    float biolumIntensity = clamp(algaeThickness.r, 0.0f, 1.0f);
    vec3 biolumColour = abs(biolumAtten * biolumIntensity) * uBioColour;


    //------------------------------------------------------------
    // Get position
    vec3 pos = ScreenToWorld(fUV, depth.r);


    //------------------------------------------------------------
    // Get normal
    float h = 0.005f;

    vec3 ddx = ScreenToWorld(fUV + vec2(h, 0.0f), texture2D(uDepthTex, fUV + vec2(h, 0.0f)).r) - pos;
    vec3 ddx2 = pos - ScreenToWorld(fUV + vec2(-h, 0.0f), texture2D(uDepthTex, fUV + vec2(-h, 0.0f)).r);
    if(abs(ddx.z) > abs(ddx2.z))
    {
        ddx = ddx2;
    }

    vec3 ddy = ScreenToWorld(fUV + vec2(0.0f, h), texture2D(uDepthTex, fUV + vec2(0.0f, h)).r) - pos;
    vec3 ddy2 = pos - ScreenToWorld(fUV + vec2(0.0f, -h), texture2D(uDepthTex, fUV + vec2(0.0f, -h)).r);
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
    // Get reflected ray and colour
    vec3 reflectRay = reflect(-eye, normal);
    vec3 reflectColour = textureCube(uCubeMapTex, reflectRay).rgb;


    //------------------------------------------------------------
    // Get refracted ray and colour
    vec3 refractRay = refract(-eye, normal, 1.333f);
    vec3 refractColour = textureCube(uCubeMapTex, refractRay).rgb;
    float attenuation = clamp(thickness.r, 0.0f, 1.0f);


    //------------------------------------------------------------
    // Get frenel
    float fresnelReflect = Fresnel(-eye, normal, 1.333f);
    float fresnelRefract = 1.0f - fresnelReflect;


    //------------------------------------------------------------
    // Diffuse and Spec Shading
    vec3 diffuse = vec3(1.0f) * dot(normal, light);
    vec3 specular = vec3(0.3f) * pow(clamp(dot(reflectRay, light), 0.0f, 1.0f), 4);


    //------------------------------------------------------------
    // final colour
    fragColor.rgb = (fresnelReflect * diffuse * reflectColour) + ((1.0f - attenuation) * fresnelRefract * refractColour) + (attenuation * uFluidColour) + specular + biolumColour;

    fragColor.a = (attenuation);
    gl_FragDepth = depth.g;


    //------------------------------------------------------------
    // Debug shading
//    fragColor.rgb = normal;
//    fragColor.rgb = diffuse;
//    fragColor.rgb = specular;
//    fragColor.rgb = reflectColour;
//    fragColor.rgb = refractColour;
//    fragColor.rgb = vec3(depth.b);
//    fragColor.rgb = thickness.rgb;
//    fragColor.rgb = algaeThickness.rgb;

}
