#version 140

in vec2 v_texCoord;
out float o_pixel;

uniform sampler2D intermediateTexture;
uniform float bias;

void main()
{
	float s = texture2D(intermediateTexture, v_texCoord).r;
	s += bias;
	s = max(s, 0) + min(s, 0) * 0.1;
	o_pixel = s;
}
