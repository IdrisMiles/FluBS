#version 330

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 gPos[];
in vec3 gVel[];
in float gDen[];

out vec3 fPos;
out vec3 fVel;
out float fDen;
out vec2 fTexCoord;

uniform mat4 uProjMatrix;
uniform mat4 uMVMatrix;
uniform vec3 uCameraPos;
uniform float uRad;

void main()
{

    vec3 Pos = gl_in[0].gl_Position.xyz;

    vec3 toCamera = normalize(uCameraPos - Pos);
    vec3 upTmp = vec3(0, 1, 0);

    // Have to do some dodgy corrections
    if(abs(dot(toCamera,upTmp)) > 0.5f)
    {
        upTmp = vec3(0,0,1);
    }
    if(abs(dot(toCamera,upTmp)) > 0.5f)
    {
        upTmp = vec3(1,0,0);
    }

    vec3 right = cross(toCamera, upTmp);
    vec3 up = cross(right, toCamera);

    right *= 2*uRad;
    up *= 2*uRad;

    // Bottom left
    Pos -= (right * 0.5);
    Pos -= (up * 0.5);
    gl_Position = uProjMatrix * uMVMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(0.0, 0.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    // Top left
    Pos += up;
    gl_Position = uProjMatrix * uMVMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(0.0, 1.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    // Bottom right
    Pos -= up;
    Pos += right;
    gl_Position = uProjMatrix * uMVMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(1.0, 0.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    // Top right
    Pos += up;
    gl_Position = uProjMatrix * uMVMatrix * vec4(Pos, 1.0);
    fTexCoord = vec2(1.0, 1.0);
    fPos = gPos[0];
    fVel = gVel[0];
    fDen = gDen[0];
    EmitVertex();

    EndPrimitive();

}
