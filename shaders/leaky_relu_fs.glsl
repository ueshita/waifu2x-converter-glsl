#version 140

precision highp float;

varying vec2 v_texCoord;

uniform sampler2D intermediateTexture;

uniform float bias;

void main()
{
	float s = texture2D(intermediateTexture, v_texCoord).r;
	s += bias;
	s = max(s, 0) + min(s, 0) * 0.1;
	gl_FragColor = vec4(s, 0, 0, 1);
}
