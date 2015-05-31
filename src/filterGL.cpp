
#include <fstream>
#include <sstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "filterGL.h"

#define	CHECK_GL_ERROR	assert(glGetError() == 0)

struct FilterVertex
{
	float x, y;
	float tu, tv;
};

static GLFWwindow* window = nullptr;
static GLuint intermediateTexture = 0;
static GLuint fboConvolve = 0;
static GLuint outputTexture = 0;
static GLuint fboLeakyReLU = 0;
static GLuint pboOutput = 0;

// 畳み込み処理シェーダ―
static struct {
	GLuint program;
	GLuint a_position;
	GLuint a_texCoord;
	
	GLuint inputTexture;
	GLuint inputDelta;
	GLuint weightMatrix;
} sConvoluve = {};

// Leaky ReLU処理シェーダ―
static struct {
	GLuint program;
	GLuint a_position;
	GLuint a_texCoord;
	
	GLuint intermediateTexture;
	GLuint bias;
} sLeakyReLU;

static std::string loadFile(const char *path);
static bool loadShader(const char *vsPath, const char *fsPath, GLuint *prog);

void filterGLInit(uint32_t width, uint32_t height)
{
	glfwInit();

	glfwWindowHint(GLFW_VISIBLE, 0);
	window = glfwCreateWindow(1, 1, "waifu2x-glsl", nullptr, nullptr);
	assert(window);

	glfwMakeContextCurrent(window);
	
	GLenum glewResult = glewInit();
	assert(glewResult == 0);

	if (!loadShader("shaders/convolve_vs.glsl", "shaders/convolve_fs.glsl", &sConvoluve.program)) {
		std::cout << "shader compile error." << std::endl;
		return;
	}
	sConvoluve.a_position = glGetAttribLocation(sConvoluve.program, "a_position");
	sConvoluve.a_texCoord = glGetAttribLocation(sConvoluve.program, "a_texCoord");
	sConvoluve.inputTexture = glGetUniformLocation(sConvoluve.program, "inputTexture");
	sConvoluve.inputDelta   = glGetUniformLocation(sConvoluve.program, "inputDelta");
	sConvoluve.weightMatrix = glGetUniformLocation(sConvoluve.program, "weightMatrix");

	if (!loadShader("shaders/leaky_relu_vs.glsl", "shaders/leaky_relu_fs.glsl", &sLeakyReLU.program)) {
		std::cout << "shader compile error." << std::endl;
		return;
	}
	sLeakyReLU.a_position = glGetAttribLocation(sLeakyReLU.program, "a_position");
	sLeakyReLU.a_texCoord = glGetAttribLocation(sLeakyReLU.program, "a_texCoord");
	sLeakyReLU.intermediateTexture = glGetUniformLocation(sLeakyReLU.program, "intermediateTexture");
	sLeakyReLU.bias                = glGetUniformLocation(sLeakyReLU.program, "bias");
	
	glGenTextures(1, &intermediateTexture);
	glBindTexture(GL_TEXTURE_2D, intermediateTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	CHECK_GL_ERROR;
	
	glGenFramebuffers(1, &fboConvolve);
	glBindFramebuffer(GL_FRAMEBUFFER, fboConvolve);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, intermediateTexture, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	CHECK_GL_ERROR;
	
	glGenTextures(1, &outputTexture);
	glBindTexture(GL_TEXTURE_2D, outputTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	CHECK_GL_ERROR;

	glGenFramebuffers(1, &fboLeakyReLU);
	glBindFramebuffer(GL_FRAMEBUFFER, fboLeakyReLU);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTexture, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CHECK_GL_ERROR;
	
	glGenBuffers(1, &pboOutput);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pboOutput);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CHECK_GL_ERROR;
}

void filterGLRelease()
{
	glDeleteFramebuffers(1, &fboLeakyReLU);
	glDeleteFramebuffers(1, &fboConvolve);
	glDeleteTextures(1, &outputTexture);
	glDeleteTextures(1, &intermediateTexture);

	glfwTerminate();
}

bool filterGLProcess(std::vector<cv::Mat> &inputPlanes,
		std::vector<cv::Mat> &weightMatrices, std::vector<double> &biases,
		std::vector<cv::Mat> &outputPlanes)
{
	cv::Size ipSize = inputPlanes[0].size();

	glEnable(GL_TEXTURE_2D);

	GLuint textures[256] = {};
	glGenTextures(inputPlanes.size(), textures);
	for (size_t i = 0; i < inputPlanes.size(); i++) {
		void *pixels = inputPlanes[i].data;
		size_t size = (size_t)(inputPlanes[i].dataend - inputPlanes[i].data);
		
		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, ipSize.width, ipSize.height, 0, GL_RED, GL_FLOAT, pixels);
	}
	CHECK_GL_ERROR;
	
	const FilterVertex vertices[4] = {
		{-1.0f,  1.0f, 0.0f, 1.0f},
		{-1.0f, -1.0f, 0.0f, 0.0f},
		{ 1.0f, -1.0f, 1.0f, 0.0f},
		{ 1.0f,  1.0f, 1.0f, 1.0f},
	};

	for (size_t opIndex = 0; opIndex < outputPlanes.size(); opIndex++) {
		glBindFramebuffer(GL_FRAMEBUFFER, fboConvolve);
		glViewport(0, 0, ipSize.width, ipSize.height);

		glUseProgram(sConvoluve.program);
		
		glEnableVertexAttribArray(sConvoluve.a_position);
		glVertexAttribPointer(sConvoluve.a_position, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), &vertices->x);
		glEnableVertexAttribArray(sConvoluve.a_texCoord);
		glVertexAttribPointer(sConvoluve.a_texCoord, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), &vertices->tu);
		
		glUniform2f(sConvoluve.inputDelta,  1.0f / ipSize.width, 1.0f / ipSize.height);
		
		for (size_t ipIndex = 0; ipIndex < inputPlanes.size(); ipIndex++) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textures[ipIndex]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glUniform1i(sConvoluve.inputTexture, 0);

			auto& weightMatrix = weightMatrices[opIndex * inputPlanes.size() + ipIndex];
			glUniform3fv(sConvoluve.weightMatrix, 3, (float*)weightMatrix.data);

			if (ipIndex == 0) {
				glDisable(GL_BLEND);
			} else if (ipIndex == 1) {
				glEnable(GL_BLEND);
				glBlendFunc(GL_ONE, GL_ONE);
			}
			glDrawArrays(GL_QUADS, 0, 4);
		}
		CHECK_GL_ERROR;
		//glFlush();
		//glFinish();
		
		glBindFramebuffer(GL_FRAMEBUFFER, fboLeakyReLU);
		glViewport(0, 0, ipSize.width, ipSize.height);

		glUseProgram(sLeakyReLU.program);
		
		glEnableVertexAttribArray(sLeakyReLU.a_position);
		glVertexAttribPointer(sLeakyReLU.a_position, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), &vertices->x);
		glEnableVertexAttribArray(sLeakyReLU.a_texCoord);
		glVertexAttribPointer(sLeakyReLU.a_texCoord, 2, GL_FLOAT, GL_FALSE, sizeof(FilterVertex), &vertices->tu);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, intermediateTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glUniform1i(sLeakyReLU.intermediateTexture, 0);
		glUniform1f(sLeakyReLU.bias, (float)biases[opIndex]);
		
		glDisable(GL_BLEND);
		glDrawArrays(GL_QUADS, 0, 4);
		
		CHECK_GL_ERROR;
		//glFlush();
		//glFinish();
		
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pboOutput);
		glReadPixels(0, 0, ipSize.width, ipSize.height, GL_RED, GL_FLOAT, 0);
		{
			auto& matrix = outputPlanes[opIndex];
			void *pixels = matrix.data;
			size_t size = (size_t)(matrix.dataend - matrix.data);
			
			void *resultAddr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
			memcpy(pixels, resultAddr, ipSize.width * ipSize.height * sizeof(float));
			glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
		}
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	glDeleteTextures(inputPlanes.size(), textures);

	return true;
}

static bool loadFile(const char *path, std::string& outstr)
{
	std::ifstream file;
	file.open(path);
	if (!file.is_open()) {
		return false;
	}
	std::istreambuf_iterator<char> it(file);
	std::istreambuf_iterator<char> last;
	outstr = std::string(it, last);
	return true;
}

static bool loadShader(const char *vsPath, const char *fsPath, GLuint *prog)
{
	std::string vsCode, fsCode;
	
	if (!loadFile(vsPath, vsCode)) return false;
	if (!loadFile(fsPath, fsCode)) return false;

	GLint linked, compiled;
	GLuint vsh = glCreateShader(GL_VERTEX_SHADER);
	GLuint fsh = glCreateShader(GL_FRAGMENT_SHADER);

	const char *vsCodePtr = vsCode.c_str();
	const char *fsCodePtr = fsCode.c_str();
	
	glShaderSource(vsh, 1, &vsCodePtr, 0);
	glCompileShader(vsh);
	glGetShaderiv(vsh, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_FALSE) {
		char log[512];
		glGetShaderInfoLog(vsh, sizeof(log), 0, log);
		std::cout << log << std::endl;
	    return false;
	}

	glShaderSource(fsh, 1, &fsCodePtr, 0);
	glCompileShader(fsh);
	glGetShaderiv(fsh, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_FALSE) {
		char log[512];
		glGetShaderInfoLog(fsh, sizeof(log), 0, log);
		std::cout << log << std::endl;
	    return false;
	}

	*prog = glCreateProgram();
	glAttachShader(*prog, vsh);
	glAttachShader(*prog, fsh);
	glLinkProgram(*prog);

	glDeleteShader(vsh);
	glDeleteShader(fsh);
	
	glGetProgramiv(*prog, GL_LINK_STATUS, &linked);
	if (linked == GL_FALSE) {
		return false;
	}

	return true;
}
