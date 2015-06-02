#version 140

in vec2 v_texCoord;
out float o_pixel;

uniform sampler2D inputTexture;
uniform vec3 weightMatrix[3];

void main()
{
	highp vec3 s0 = vec3(textureOffset(inputTexture, v_texCoord, ivec2(-1, -1)).r,
	                     textureOffset(inputTexture, v_texCoord, ivec2( 0, -1)).r,
	                     textureOffset(inputTexture, v_texCoord, ivec2( 1, -1)).r);
	highp vec3 s1 = vec3(textureOffset(inputTexture, v_texCoord, ivec2(-1,  0)).r,
	                     texture      (inputTexture, v_texCoord               ).r,
	                     textureOffset(inputTexture, v_texCoord, ivec2( 1,  0)).r);
	highp vec3 s2 = vec3(textureOffset(inputTexture, v_texCoord, ivec2(-1,  1)).r,
	                     textureOffset(inputTexture, v_texCoord, ivec2( 0,  1)).r,
	                     textureOffset(inputTexture, v_texCoord, ivec2( 1,  1)).r);

	highp float s = dot(s0, weightMatrix[0]) +
	                dot(s1, weightMatrix[1]) +
	                dot(s2, weightMatrix[2]);
	
	o_pixel = s;
}
