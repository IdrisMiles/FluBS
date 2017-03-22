#version 430

in vec2 fUV;

uniform sampler2D uDepthTex;

out vec4 fragColor;

float Weight(float _i, float _ni, float _dx, float _dy, float _smoothR, float _smoothS)
{
    float ri = abs(_i - _ni);
    float rangeKernel = -(ri*ri)/(2.0f*(_smoothR*_smoothR));

    float spatialKernel = -((_dx*_dx)+(_dy*_dy))/(2.0f*(_smoothS*_smoothS));

    return exp(spatialKernel + rangeKernel);
}

float BilateralFilter(float _i, float _rad, int _dim, float _smoothR, float _smoothS)
{
    float totalWeight = 0.0f;
    float totalDepth = 0.0f;
    for(int x=0; x< _dim; x++)
    {
        for(int y=-_dim; y< _dim; y++)
        {
            float dx = (x * _rad);
            float dy = (y * _rad);
            float neighDepth = texture2D(uDepthTex, fUV + vec2(dx, dy)).r;

            float weight = Weight(_i, neighDepth, dx, dy, _smoothR, _smoothS);

            totalDepth += weight * neighDepth;
            totalWeight += weight;
        }
    }

    return totalDepth / totalWeight;
}

void main()
{
    vec4 depth = texture2D(uDepthTex, fUV.xy);
    if(depth.a < 0.1f)
    {
        discard;
    }


    fragColor.xyz = vec3(depth.r, depth.g, BilateralFilter(depth.b, 0.005, 10, 0.02, 0.2));
    fragColor.a = 1.0f;
}
