#version 430

in vec2 fUV;

uniform vec3 uCameraPos;

uniform sampler2D uDepthTex;
uniform sampler2D uThicknessTex;
uniform samplerCube uCubeMapTex;

out vec4 fragColor;

void main()
{
    vec4 depth = texture(uDepthTex, fUV);
    vec4 thickness = texture(uThicknessTex, fUV);
    if(depth.a <= 0.1f || thickness.r <= 0.0f)
    {
        discard;
    }

    float h = 0.002f;
    float h2 = 2.0f*h;

    // Get position
    float px = (fUV.x - 0.5f) * 2.0f;
    float py = (fUV.y - 0.5f) * 2.0f;
    vec3 pos = vec3(px, py, depth.r);

    // Get normal
    float nx = (texture(uDepthTex, fUV + vec2(h, 0.0f)).r - texture(uDepthTex, fUV + vec2(-h, 0.0f)).r) / h2;
    float ny = (texture(uDepthTex, fUV + vec2(0.0f, h)).r - texture(uDepthTex, fUV + vec2(0.0f, -h)).r) / h2;
//    float z2 = 1.0f - ((nx*nx)+(ny*ny));
//    float z = sqrt(abs(z2));
//    z *= sign(z2);
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
    refractColour *= (1.0f - attenuation);



    // final colour
    fragColor.rgb = refractColour + (vec3(0.1f,0.4f,0.9f) * attenuation) + (reflectColour*0.4f);
//    fragColor.rgb = depth.rgb;
//    fragColor.rgb = thickness.rgb;
}
