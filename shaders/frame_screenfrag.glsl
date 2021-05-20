#version 320 es

in mediump vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D screenTexture;

void main()
{ 	
	highp vec3 color = vec3(texture(screenTexture, textCoord)).rgb;
	

	FragColor = vec4(color, 1.0f);
}
