#version 320 es

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec2 vTexCoord;



uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


out vec2 textCoord;

void main()
{
    gl_Position = projection * view * model * vec4(vPos.x, vPos.y, vPos.z, 1.f);
    textCoord = vec2(vTexCoord.x, 1.f - vTexCoord.y);
}
