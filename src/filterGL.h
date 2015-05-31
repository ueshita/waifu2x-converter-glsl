
#ifndef FILTER_GL_H_
#define FILTER_GL_H_

#include <stdint.h>
#include <opencv2/opencv.hpp>

void filterGLInit(uint32_t width, uint32_t height);
void filterGLRelease();
bool filterGLProcess(std::vector<cv::Mat> &inputPlanes,
		std::vector<cv::Mat> &weightMatrices, std::vector<double> &biases,
		std::vector<cv::Mat> &outputPlanes);

#endif