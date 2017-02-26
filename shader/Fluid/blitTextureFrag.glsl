#version 330

in vec2 fUV;

uniform sampler2D uTex;

out vec4 fragColor;


void main()
{
    vec4 colour = texture(uTex, fUV.xy);

    if(colour.a < 0.1f)
    {
        discard;
    }

    fragColor.rgb = colour.rgb;
}
