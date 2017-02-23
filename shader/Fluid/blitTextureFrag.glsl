#version 330

in vec2 fUV;

uniform sampler2D uTex;

out vec4 fragColor;


void main()
{
    fragColor.rgb = texture(uTex, fUV.xy).rgb;
}
