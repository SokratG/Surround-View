#version 320 es

in mediump vec2 texCoord;
in mediump vec3 normal;


out highp vec4 FragColor;

uniform lowp vec3 Ka;
uniform lowp vec3 Kd;
uniform lowp vec3 Ks;
uniform lowp float shininess;


uniform sampler2D texture_diffuse1;
//uniform sampler2D texture_specular1;

highp vec4 colorModel()
{
    // TODO - add light and material computation
    highp vec4 red_color = vec4(0.5f, 0.f, 0.f, 1.f);
    return red_color * vec4((Ka + Kd + Ks), 1.f);
}

void main()
{  
        mediump vec4 color_diff = texture(texture_diffuse1, texCoord);
        FragColor = colorModel();
}
