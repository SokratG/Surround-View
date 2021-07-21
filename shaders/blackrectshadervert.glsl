#version 320 es

layout (location = 0) in vec3 vPos;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main()
{
	gl_Position = projection * view * model * vec4(vPos.x, vPos.y, vPos.z, 1.f);
}
