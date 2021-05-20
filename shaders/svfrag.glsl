#version 320 es

in mediump vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D surroundTexture;

const lowp float gamma_coef = 2.2f;

const lowp float alpha = 0.3f;

highp vec3 compute_tex_mix()
{
	highp vec3 color = vec3(texture(surroundTexture, textCoord));
	
	// blend last and first frame
	if (textCoord.x >= 0.999f){
		mediump vec2 tex_cord = vec2(1.f - textCoord.x, textCoord.y); 
		color = mix(color, vec3(texture(surroundTexture, tex_cord)), alpha);
	}

	return color;
}


void main()
{ 
	

	highp vec3 color = compute_tex_mix();

	color = pow(color, vec3(1.f / gamma_coef)); 
	
	highp vec4 color_a = vec4(color, 1.f);
 
	if (textCoord.x > 1.0f || textCoord.y > 1.0f )
		color_a.a = 0.0f;

	FragColor = color_a;
}
