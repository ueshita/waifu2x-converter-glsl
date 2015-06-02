
in vec2 v_texCoord;

out float o_pixel;

uniform float bias;
uniform sampler2DArray inputTextures;
uniform vec3 weightMatrix[3 * 128];

void main()
{
	// Convolution Process
	highp float s = 0.0;
	for (int i = 0; i < NUM_INPUT_PLANES; i++) {
		highp vec3 t0, t1, t2;
		vec3 uvt = vec3(v_texCoord, i);
		t0 = vec3(textureOffset(inputTextures, uvt, ivec2(-1, -1)).r,
		          textureOffset(inputTextures, uvt, ivec2( 0, -1)).r,
		          textureOffset(inputTextures, uvt, ivec2( 1, -1)).r);
		t1 = vec3(textureOffset(inputTextures, uvt, ivec2(-1,  0)).r,
		          texture      (inputTextures, uvt               ).r,
		          textureOffset(inputTextures, uvt, ivec2( 1,  0)).r);
		t2 = vec3(textureOffset(inputTextures, uvt, ivec2(-1,  1)).r,
		          textureOffset(inputTextures, uvt, ivec2( 0,  1)).r,
		          textureOffset(inputTextures, uvt, ivec2( 1,  1)).r);
		
		s += dot(t0, weightMatrix[i * 3 + 0]) +
	         dot(t1, weightMatrix[i * 3 + 1]) +
	         dot(t2, weightMatrix[i * 3 + 2]);
	}
	
	// Leaky ReLU Process
	s += bias;
	s = max(s, 0) + min(s, 0) * 0.1;
	o_pixel = s;
}
