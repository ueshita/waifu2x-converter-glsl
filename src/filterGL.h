
#ifndef FILTER_GL_H_
#define FILTER_GL_H_

#include <stdint.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

// waifu2x shader
struct Waifu2xShader
{
	GLuint program;
	GLuint a_position;
	GLuint a_texCoord;
	
	GLuint bias;
	GLuint weightMatrix;
	GLuint inputTextures;
};

void filterGLInit(uint32_t width, uint32_t height);

void filterGLRelease();

void filterGLSetInputData(cv::Mat& inputPlane);

void filterGLGetOutputData(cv::Mat& outputPlane);

bool filterGLProcess(Waifu2xShader& shader, 
	int nInputPlanes, int nOutputPlanes,
	std::vector<cv::Mat> &weightMatrices, 
	std::vector<double> &biases, int modelIndex);

#endif
