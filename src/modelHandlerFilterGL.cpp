
#include "modelHandler.hpp"
#include "filterGL.h"

bool w2xc::Model::filterGL(std::vector<cv::Mat> &inputPlanes,
	std::vector<cv::Mat> &outputPlanes)
{
	
	if (inputPlanes.size() != nInputPlanes) {
		std::cerr << "Error : Model-filter : \n"
				"number of input planes mismatch." << std::endl;
		std::cerr << inputPlanes.size() << ","
				<< nInputPlanes << std::endl;
		return false;
	}

	outputPlanes.clear();
	for (int i = 0; i < nOutputPlanes; i++) {
		outputPlanes.push_back(cv::Mat::zeros(inputPlanes[0].size(), CV_32FC1));
	}

	// filter job issuing
	filterGLProcess(inputPlanes, weights, biases, outputPlanes);

	return true;
}
