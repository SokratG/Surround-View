#version 320 es

in highp vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D surroundTexture;

const lowp float gamma_coef = 2.2f;
const lowp float blend_factor = 1.75f;


uniform mediump float t_intensity;
uniform mediump float light_adapt;
uniform mediump float color_adapt;
uniform mediump float map_key;
uniform mediump float gray_mean;

uniform mediump vec3 chan_mean;


highp vec3 gamma_correction(const highp vec3 color, const lowp float g_coef)
{	
	return pow(color, vec3(1.f / g_coef)); 
}

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


// Reinhard algorithm
highp float Reinhard_tonemap(highp float gray, highp float chan_mean_, 
			    const highp float color_chan)
{

	highp float global = color_adapt * chan_mean_ + (1.0f - color_adapt) * gray_mean;
	
	highp float adapt = color_adapt * color_chan + (1.0f - color_adapt) * gray;

	adapt = light_adapt * adapt + (1.0f - light_adapt) * global;

	adapt = pow(t_intensity * adapt, map_key);

	highp float color_channel = color_chan * (1.0f / (adapt + color_chan));

	return color_channel;
}


highp vec3 tonemap(const highp vec3 color)
{	
	highp float gray_color = 0.229f * color.r + 0.587f * color.g + 0.114f * color.b; 
	
	highp float _red = Reinhard_tonemap(gray_color, chan_mean.r, color.r);
	highp float _green = Reinhard_tonemap(gray_color, chan_mean.g, color.g);
	highp float _blue = Reinhard_tonemap(gray_color, chan_mean.b, color.b);
	
	highp vec3 union_color = vec3(_red, _green, _blue);
	
	return union_color;
}


highp float luminance(const highp vec3 v)
{
	return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

highp vec3 change_luminance(highp vec3 c_in, highp float l_out)
{
	highp float l_in = luminance(c_in);
	return c_in * (l_out / l_in);
}

highp vec3 reinhard_luminance(highp vec3 color_in, highp float max_white)
{
	highp float l_old = luminance(color_in);
	highp float numerator = l_old * (1.0f + (l_old / (max_white * max_white)));
	highp float l_new = numerator / (1.f + l_old);
	return change_luminance(color_in, l_new);
}

highp vec3 test_tone(vec3 v)
{
	highp float l = luminance(v);
	highp vec3 tv = v / (1.f + v);
	return mix(v / (1.f + l), tv, tv);
}

void main()
{ 
	

	highp vec3 color = compute_tex_mix();

	//color = tonemap(color);
	
	color = test_tone(color);

	color = gamma_correction(color, gamma_coef);
	
	highp vec4 color_a = vec4(color, 1.f);

	FragColor = color_a;
}
