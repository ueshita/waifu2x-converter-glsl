
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <conio.h>
#include "picojson.h"

#include "modelHandler.hpp"
#include "filterGL.h"

using namespace cv;

const char *modelPath    = "models/scale2.0x_model.json";
const char *modelPathBin = "models/scale2.0x_model.bin";
const char *inputPath  = "256.jpg";
const char *outputPath = "out.jpg";

int main(int argc, char** argv)
{
	std::vector<std::unique_ptr<w2xc::Model> > models;

	//w2xc::modelUtility::generateModelFromJSON(modelPath, models);
	//w2xc::modelUtility::saveModelToBin(modelPathBin, models);
	//models.clear();
	w2xc::modelUtility::generateModelFromBin(modelPathBin, models);
	
	std::cout << "reading model data seems to be succeed." << std::endl;

	cv::Mat image   = cv::imread(inputPath, cv::IMREAD_COLOR);
	cv::Mat image2x = cv::Mat(image.size().height * 2, image.size().width * 2, CV_32FC3);
	cv::resize(image,image2x,image2x.size(),0,0,INTER_NEAREST);
	cv::imwrite("tmp_nearest.png",image2x);
	
	cv::Mat imageYUV;
	image2x.convertTo(imageYUV, CV_32F, 1.0 / 255.0);
	cv::cvtColor(imageYUV, imageYUV, COLOR_RGB2YUV);
	std::vector<cv::Mat> imageSprit;
	
	cv::Mat imageY;
	cv::split(imageYUV, imageSprit);
	imageSprit[0].copyTo(imageY);

	imageSprit.clear();
	cv::Mat image2xBicubic;
	cv::resize(image,image2xBicubic,image2x.size(),0,0,INTER_CUBIC);
	cv::imwrite("tmp_bicubic.png",image2xBicubic);
	image2xBicubic.convertTo(imageYUV,CV_32F, 1.0 / 255.0);
	cv::cvtColor(imageYUV, imageYUV, COLOR_RGB2YUV);
	cv::split(imageYUV, imageSprit);
	
	std::unique_ptr<std::vector<cv::Mat> > inputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>(1));
	std::unique_ptr<std::vector<cv::Mat> > outputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>(32));

	inputPlanes->clear();
	inputPlanes->push_back(imageY);

	auto size = image2x.size();
	filterGLInit(size.width, size.height);

	auto start = std::chrono::system_clock::now();

	for (int index = 0; index < (int)models.size(); index++) {
		std::cout << "Iteration #" << (index + 1) << "..." << std::endl;
		
		//std::cout << 
		//	"input:" << inputPlanes->size() << 
		//	",output:" << outputPlanes->size() << std::endl;
		
		//if(!models[index]->filter(*inputPlanes, *outputPlanes)){
		if(!models[index]->filterGL(*inputPlanes, *outputPlanes)){
			std::exit(-1);
		}
		
		if (index != models.size() - 1) {
			inputPlanes = std::move(outputPlanes);
			outputPlanes = std::unique_ptr<std::vector<cv::Mat> >(
					new std::vector<cv::Mat>(models[index + 1]->getNOutputPlanes()));
		}
	}
	
	auto end = std::chrono::system_clock::now();
	auto diff = end - start;
	std::cout << "elapsed time = "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
			  << " msec."
			  << std::endl;

	filterGLRelease();

	outputPlanes->at(0).copyTo(imageSprit[0]);
	cv::Mat result;
	cv::merge(imageSprit,result);
	cv::cvtColor(result,result,COLOR_YUV2RGB);
	result.convertTo(result,CV_8U,255.0);
	cv::imwrite(outputPath,result);

	return 0;
}

