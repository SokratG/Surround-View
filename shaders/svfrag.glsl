#version 320 es

in highp vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D surroundTexture;

const lowp float gamma_coef = 2.2f;
const lowp float blend_factor = 1.75f;


highp vec3 compute_tex_mix()
{
	highp vec3 color = vec3(texture(surroundTexture, textCoord));
	
	// blend last and first frame
	if (textCoord.x >= 1.0f){
		highp vec2 tex_cord = vec2(0.0f, textCoord.y); 
		// weight coefficient for blending 
		mediump float alpha = textCoord.x / blend_factor; 
		color = mix(color, vec3(texture(surroundTexture, tex_cord)), alpha);
	}

	if (textCoord.x <= 0.0f){
		highp vec2 tex_cord = vec2(1.0f, textCoord.y); 
		// weight coefficient for blending 
		mediump float alpha = tex_cord.x / blend_factor; 
		color = mix(color, vec3(texture(surroundTexture, tex_cord)), alpha);
	}

	return color;
}


void main()
{ 
	

	highp vec3 color = compute_tex_mix();

	color = pow(color, vec3(1.f / gamma_coef)); 
	
	highp vec4 color_a = vec4(color, 1.f);

	FragColor = color_a;
}
