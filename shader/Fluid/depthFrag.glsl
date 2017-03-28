#version 430

in vec3 fVel;
in float fDen;
in vec3 fPos;
in vec2 fTexCoord;

layout (location = 0) out vec4 oDepth;

uniform float near = 0.1f;
uniform float far = 30.0f;
uniform float uRestDen = 1000.0f;
uniform float uRad = 0.2f;
uniform mat4 uProjMatrix;

float LinearizeDepth(float depth, float near, float far)
{
    // Back to NDC
    float z = depth * 2.0 - 1.0;
    // Make linear
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main()
{
    if(fDen < 0.5f*uRestDen)
    {
//        discard;
//        return;
    }


    float x = 2.0f * (fTexCoord.x - 0.5f);
    float y = 2.0f * (fTexCoord.y - 0.5f);
    float z2 = ((x*x)+(y*y));


    if(z2 > 1.0f)
    {
        discard;
    }

    float z = sqrt(1.0f - z2);
    vec4 pixelPos = vec4(fPos + (vec3(x, y, z)*uRad), 1.0f);
    vec4 clipSpacePos = uProjMatrix * pixelPos;
    float depth = clipSpacePos.z / clipSpacePos.w;
    gl_FragDepth = depth;
    float invW = 1.0f/clipSpacePos.w;

//    oDepth = vec4(vec3((LinearizeDepth(gl_FragCoord.z, near, far)+(z*uRad))/far), 1.0);
//    oDepth = vec4(vec3(LinearizeDepth(gl_FragDepth, near, far)/(far-near)), 1.0);
    oDepth = vec4(vec3(depth, gl_FragDepth, depth), 1.0);

}
