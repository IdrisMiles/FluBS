#version 430

in vec3 fVel;
in float fDen;
in vec2 fTexCoord;

layout (location = 0) out vec4 oDepth;

uniform float near = 0.1f;
uniform float far = 30.0f;
uniform float uRestDen = 1000.0f;

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
        discard;
        return;
    }

    float x = 2.0f * (fTexCoord.x - 0.5f);
    float y = 2.0f * (fTexCoord.y - 0.5f);
    float z2 = 1.0 - ((x*x)+(y*y));
    float z = 0.0f;

    if(z2 < 0.0f)
    {
        discard;
        return;
    }

    z = sqrt(z2) / 10.0f;

    oDepth = vec4(vec3((LinearizeDepth(gl_FragCoord.z, near, far)/*-z*/)/far), 1.0);

}
