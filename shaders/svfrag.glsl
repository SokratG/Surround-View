#version 320 es

in highp vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D useTexture;

const lowp float gamma_coef = 2.2f;

void main()
{ 
	highp vec3 color = vec3(texture(useTexture, textCoord));
        // color = pow(color, vec3(1.f / gamma_coef));
	FragColor = vec4(color, 1.f);
}
