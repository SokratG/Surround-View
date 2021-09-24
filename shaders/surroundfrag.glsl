#version 320 es

in highp vec2 textCoord;

out highp vec4 FragColor;

uniform sampler2D surroundTexture;

const lowp float gamma_coef = 2.2f;
const lowp float blend_factor = 1.75f;
const lowp vec3 lum_vec = vec3(0.2126f, 0.7152f, 0.0722f);
const lowp float lum_thresh_white = 0.85f;

uniform highp float lum_white;
uniform highp float lum_map;

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


// Reinhard tonemap algorithm
highp float luminance(const highp vec3 v)
{
	return dot(v, vec3(lum_map, lum_map, lum_map));
}

highp vec3 change_luminance(const highp vec3 c_in, const highp float l_out)
{
	highp float l_in = luminance(c_in);
	return c_in * (l_out / l_in);
}

highp vec3 reinhard_tonemap(highp vec3 color_in, const highp float max_white)
{
	highp float l_old = luminance(color_in);
	highp float numerator = l_old * (1.0f + (l_old / (max_white * max_white)));
	highp float l_new = numerator / (1.f + l_old);
	return change_luminance(color_in, l_new);
}

highp vec3 reinhard_jodie_tonemap(vec3 c_in)
{
	highp float l = luminance(c_in);
	highp vec3 tv = c_in / (1.f + c_in);
	return clamp(mix(c_in / (1.f + l), tv, tv), 0.0f, 1.0f);
}

void main()
{

	highp vec3 color = compute_tex_mix();
	
        highp float white_luminance = lum_white > lum_thresh_white ? lum_thresh_white : lum_white;

        color = reinhard_tonemap(color, white_luminance);

        color = gamma_correction(color, gamma_coef);
	
	highp vec4 color_a = vec4(color, 1.f);

	FragColor = color_a;
}
