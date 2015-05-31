#version 140

precision highp float;
precision highp vec3;

varying vec2 v_texCoord;

uniform sampler2D inputTexture;

uniform vec2 inputDelta;
uniform vec3 weightMatrix[3];

void main()
{
	vec2 ip = v_texCoord;
	vec2 id = inputDelta;

	vec3 s0 = vec3(texture2D(inputTexture, ip + vec2(-id.x, -id.y)).r,
				   texture2D(inputTexture, ip + vec2(    0, -id.y)).r,
				   texture2D(inputTexture, ip + vec2(+id.x, -id.y)).r);
	vec3 s1 = vec3(texture2D(inputTexture, ip + vec2(-id.x,     0)).r,
				   texture2D(inputTexture, ip + vec2(    0,     0)).r,
				   texture2D(inputTexture, ip + vec2(+id.x,     0)).r);
	vec3 s2 = vec3(texture2D(inputTexture, ip + vec2(-id.x, +id.y)).r,
				   texture2D(inputTexture, ip + vec2(    0, +id.y)).r,
				   texture2D(inputTexture, ip + vec2(+id.x, +id.y)).r);

	float s = dot(s0, weightMatrix[0]) +
			  dot(s1, weightMatrix[1]) +
			  dot(s2, weightMatrix[2]);

	gl_FragColor = vec4(s, 0, 0, 1);
}
