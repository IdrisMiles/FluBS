#version 430

in vec2 fUV;

uniform sampler2D uTex;
uniform sampler2D uDepthTex;
uniform sampler2D uThicknessTex;

out vec4 fragColor;

void main()
{
    float h = 0.01f;
    vec3 d = vec3(0.0f);

    d += texture(uTex, fUV.xy + vec2(h, 0.0f)).rgb;
    d += texture(uTex, fUV.xy + vec2(-h, 0.0f)).rgb;
    d += texture(uTex, fUV.xy + vec2(0.0f, h)).rgb;
    d += texture(uTex, fUV.xy + vec2(0.0f, -h)).rgb;
    d += texture(uTex, fUV.xy + vec2(h, h)).rgb;
    d += texture(uTex, fUV.xy + vec2(-h, h)).rgb;
    d += texture(uTex, fUV.xy + vec2(h, -h)).rgb;
    d += texture(uTex, fUV.xy + vec2(-h, -h)).rgb;

    fragColor.rgb = d / 8.0f;
}
