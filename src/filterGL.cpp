
#include <stdio.h>
#include "filterGL.h"

#define	CHECK_GL_ERROR	assert(glGetError() == 0)

struct FilterVertex
{
	float x, y;
	float tu, tv;
};

static GLFWwindow* window = nullptr;
static GLuint frameBuffer = 0;
static GLuint textureBuffers[2] = {0};
static cv::Size planeSize;

void filterGLInit(uint32_t width, uint32_t height)
{
	glfwInit();

	glfwWindowHint(GLFW_VISIBLE, 0);
	window = glfwCreateWindow(1, 1, "waifu2x-glsl", nullptr, nullptr);
	assert(window);

	glfwMakeContextCurrent(window);

	//GLint fragmentUniformVectors = 0;
	//glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS, &fragmentUniformVectors);
	
	GLenum glewResult = glewInit();
	assert(glewResult == 0);

	glGenTextures(2, textureBuffers);
	for (int i = 0; i < 2; i++) {
		glBindTexture(GL_TEXTURE_2D_ARRAY, textureBuffers[i]);
		glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32F, width, height, 128);
	}

	glGenFramebuffers(1, &frameBuffer);
	CHECK_GL_ERROR;
}

void filterGLRelease()
{
	glDeleteFramebuffers(1, &frameBuffer);
	glDeleteTextures(2, textureBuffers);

	glfwTerminate();
}

void filterGLSetInputData(cv::Mat& inputPlane)
{
	planeSize = inputPlane.size();

	glBindTexture(GL_TEXTURE_2D_ARRAY, textureBuffers[0]);
	void *pixels = inputPlane.data;
	size_t size = (size_t)(inputPlane.dataend - inputPlane.data);
	
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 
		planeSize.width, planeSize.height, 1, GL_RED, GL_FLOAT, pixels);
	CHECK_GL_ERROR;
}

void filterGLGetOutputData(cv::Mat& outputPlane)
{
	cv::Size opSize = outputPlane.size();

	glReadPixels(0, 0, opSize.width, opSize.height, GL_RED, GL_FLOAT, outputPlane.data);
}

bool filterGLProcess(Waifu2xShader& shader, 
	int nInputPlanes, int nOutputPlanes,
	std::vector<cv::Mat> &weightMatrices, 
	std::vector<double> &biases, int modelIndex)
{
	// Swap I/O double buffers
	GLuint inputTextures  = textureBuffers[(modelIndex + 0) % 2];
	GLuint outputTextures = textureBuffers[(modelIndex + 1) % 2];

	// Vertex Data
	const FilterVertex vertices[4] = {
		{-1.0f,  1.0f, 0.0f, 1.0f},
		{-1.0f, -1.0f, 0.0f, 0.0f},
		{ 1.0f, -1.0f, 1.0f, 0.0f},
		{ 1.0f,  1.0f, 1.0f, 1.0f},
	};
	
	// Temporary matrix buffer
	float vWeightMatrices[3 * 3 * 128];

	for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++) {
		glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
		glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, outputTextures, 0, opIndex);
		glViewport(0, 0, planeSize.width, planeSize.height);

		glUseProgram(shader.program);
		
		glEnableVertexAttribArray(shader.a_position);
		glVertexAttribPointer(shader.a_position, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), &vertices->x);
		glEnableVertexAttribArray(shader.a_texCoord);
		glVertexAttribPointer(shader.a_texCoord, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), &vertices->tu);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, inputTextures);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glUniform1i(shader.inputTextures, 0);
		
		glUniform1f(shader.bias, (float)biases[opIndex]);
		
		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
			memcpy(&vWeightMatrices[ipIndex * 3 * 3], 
				weightMatrices[opIndex * nInputPlanes + ipIndex].data,
				3 * 3 * sizeof(float));
		}
		glUniform3fv(shader.weightMatrix, 3 * nInputPlanes, vWeightMatrices);

		glDisable(GL_BLEND);
		glDrawArrays(GL_QUADS, 0, 4);
		CHECK_GL_ERROR;
	}

	//glFlush();
	//glFinish();

	return true;
}
