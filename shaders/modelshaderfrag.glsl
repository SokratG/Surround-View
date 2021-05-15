#version 320 es

in highp vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D texture_diffuse1;

void main()
{  
	FragColor = texture(texture_diffuse1, textCoord);
}
