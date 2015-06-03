
#include <stdio.h>
#include <exception>
#include "filterGL.h"

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

#if __APPLE__
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
	glfwWindowHint(GLFW_VISIBLE, 0);
	window = glfwCreateWindow(1, 1, "waifu2x-glsl", nullptr, nullptr);
	assert(window);

	glfwMakeContextCurrent(window);
	glGenTextures(2, textureBuffers);
	
	//std::cout << glGetString(GL_VERSION) << std::endl;
	//std::cout << glGetString(GL_VENDOR) << std::endl;
	//std::cout << glGetString(GL_EXTENSIONS) << std::endl;
	
#ifdef __glew_h__
	GLenum glewResult = glewInit();
	assert(glewResult == 0);
	glGetError();
#endif
	
	glGenTextures(2, textureBuffers);
	for (int i = 0; i < 2; i++) {
		glBindTexture(GL_TEXTURE_2D_ARRAY, textureBuffers[i]);
		glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, width, height, 128, 0, GL_RED, GL_FLOAT, nullptr);
		CHECK_GL_ERROR("glTexImage3D");
	}

	glGenFramebuffers(1, &frameBuffer);
	CHECK_GL_ERROR("glGenFramebuffers");
	
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CHECK_GL_ERROR("glBufferData");
}

void filterGLRelease()
{
	glDeleteFramebuffers(1, &frameBuffer);
	frameBuffer = 0;
	glDeleteTextures(2, textureBuffers);
	memset(textureBuffers, 0, sizeof(textureBuffers));

	glfwTerminate();
	window = nullptr;
}

void filterGLSetInputData(cv::Mat& inputPlane)
{
	planeSize = inputPlane.size();

	glBindTexture(GL_TEXTURE_2D_ARRAY, textureBuffers[0]);
	void *pixels = inputPlane.data;
	//size_t size = (size_t)(inputPlane.dataend - inputPlane.data);
	
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 
		planeSize.width, planeSize.height, 1, GL_RED, GL_FLOAT, pixels);
	CHECK_GL_ERROR("glTexSubImage3D");
}

void filterGLGetOutputData(cv::Mat& outputPlane)
{
	cv::Size opSize = outputPlane.size();
	
	GLuint outputPixelBuffer = 0;

	glGenBuffers(1, &outputPixelBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, outputPixelBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, planeSize.width * planeSize.height * sizeof(float), 0, GL_DYNAMIC_DRAW);
	CHECK_GL_ERROR("glBufferData");
	
	glReadPixels(0, 0, planeSize.width, planeSize.height, GL_RED, GL_FLOAT, 0);
	CHECK_GL_ERROR("glReadPixels");
	
	void *resultAddr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	CHECK_GL_ERROR("glMapBuffer");
	
	memcpy(outputPlane.data, resultAddr, opSize.width * opSize.height * sizeof(float));
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	
	glDeleteBuffers(1, &outputPixelBuffer);
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
		{ 1.0f,  1.0f, 1.0f, 1.0f},
		{ 1.0f, -1.0f, 1.0f, 0.0f},
	};
	
	GLuint vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	
	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(FilterVertex) * 4, vertices, GL_STATIC_DRAW);
  
	glEnableVertexAttribArray(shader.a_position);
	glVertexAttribPointer(shader.a_position, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), (void*)0);
	glEnableVertexAttribArray(shader.a_texCoord);
	glVertexAttribPointer(shader.a_texCoord, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), (void*)8);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
	glUseProgram(shader.program);
	glBindVertexArray(vao);
		
	// Temporary matrix buffer
	float vWeightMatrices[3 * 3 * 128];

	for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++) {
		glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
		glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, outputTextures, 0, opIndex);
		glViewport(0, 0, planeSize.width, planeSize.height);
		
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
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}
	CHECK_GL_ERROR("glDrawArrays");

	glBindVertexArray(0);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vao);

	//glFlush();
	glFinish();

	return true;
}
