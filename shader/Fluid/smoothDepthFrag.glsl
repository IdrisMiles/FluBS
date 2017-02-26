#version 430

in vec2 fUV;

uniform sampler2D uDepthTex;

out vec4 fragColor;

void main()
{
    vec4 depth = texture(uDepthTex, fUV.xy);
    if(depth.a < 0.1f)
    {
        discard;
    }

    float h = 0.004f;
    vec3 d = vec3(0.0f);

    d += depth.rgb;

    vec3 neighDepth;
    float range = 0.9;
    int numIterations = 20;
    for(int i=0; i<numIterations; i++)
    {
        neighDepth = texture(uDepthTex, fUV.xy + vec2(h, 0.0f)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(-h, 0.0f)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(0.0f, h)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(0.0f, -h)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(h, h)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(-h, h)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(h, -h)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;

        neighDepth = texture(uDepthTex, fUV.xy + vec2(-h, -h)).rgb;
        d += abs(neighDepth.r - depth.r) < range ? neighDepth : depth.rgb;
        h*=1.05;
    }

    fragColor = vec4(d / (float(8*numIterations)+1.0f), 1.0f);
}
