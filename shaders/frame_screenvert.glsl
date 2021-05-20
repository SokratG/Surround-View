#version 320 es

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec2 vTexCoord;


out vec2 textCoord;

void main()
{
	textCoord = vTexCoord;
	gl_Position = vec4(vPos.x, vPos.y, 0.0f, 1.0f);
}
