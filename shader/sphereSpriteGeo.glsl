#version 330

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 gPos[1];
in vec3 gVel[1];
in float gDen[1];

out vec3 fPos;
out vec3 fVel;
out float fDen;
out vec3 fTexCoord;

uniform mat4 uVPMatrix;
uniform vec3 uCameraPos;

void main()
{
    vec3 Pos = gl_in[0].gl_Position.xyz;

    vec3 toCamera = normalize(uCameraPos - Pos);
    vec3 upTmp = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(toCamera, upTmp);
    vec3 up = cross(right, toCamera);


    // Bottom left
    Pos -= (right * 0.5);
    Pos -= (up * 0.5);
    gl_Position = uVPMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(0.0, 0.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    // Top left
    Pos.y += up;
    gl_Position = uVPMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(0.0, 1.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    // Bottom right
    Pos.y -= up;
    Pos += right;
    gl_Position = uVPMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(1.0, 0.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    // Top right
    Pos.y += up;
    gl_Position = uVPMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(1.0, 1.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    EndPrimitive();

}
