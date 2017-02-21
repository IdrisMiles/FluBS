#version 330

in vec2 fUV;

uniform sampler2D tex;

out vec4 fragColor;


void main()
{
    fragColor.rgb = texture(tex, fUV.xy).rgb;
}
