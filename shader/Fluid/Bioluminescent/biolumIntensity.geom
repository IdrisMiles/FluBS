#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 gVel[];
in float gDen[];
in float gBio[];

out vec3 fVel;
out float fDen;
out vec2 fTexCoord;
out float fBio;

uniform mat4 uProjMatrix;
uniform mat4 uMVMatrix;
uniform vec3 uCameraPos;
uniform float uRad;

void main()
{

    vec3 Pos = gl_in[0].gl_Position.xyz;

    vec4 right = vec4(uRad, 0.0f, 0.0f, 0.0f);
    vec4 up = vec4(0.0f, uRad, 0.0f, 0.0f);
    vec4 viewPos = uMVMatrix * vec4(Pos, 1.0);

    // Bottom left
    gl_Position = uProjMatrix * (viewPos - up - right);
    fTexCoord = vec2(0.0, 0.0);
    fVel = gVel[0];
    fDen = gDen[0];
    fBio = gBio[0];
    EmitVertex();

    // Top left
    gl_Position = uProjMatrix * (viewPos + up - right);
    fTexCoord = vec2(0.0, 1.0);
    fVel = gVel[0];
    fDen = gDen[0];
    fBio = gBio[0];
    EmitVertex();

    // Bottom right
    gl_Position = uProjMatrix * (viewPos - up + right);
    fTexCoord = vec2(1.0, 0.0);
    fVel = gVel[0];
    fDen = gDen[0];
    fBio = gBio[0];
    EmitVertex();

    // Top right
    gl_Position = uProjMatrix * (viewPos + up + right);
    fTexCoord = vec2(1.0, 1.0);
    fVel = gVel[0];
    fDen = gDen[0];
    fBio = gBio[0];
    EmitVertex();

    EndPrimitive();

}
