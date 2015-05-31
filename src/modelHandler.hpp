
#ifndef MODEL_HANDLER_HPP_
#define MODEL_HANDLER_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "picojson.h"
#include <iostream>
#include <memory>
#include <cstdint>
#include <cstdlib>

namespace w2xc {

class Model {

private:
	int nInputPlanes;
	int nOutputPlanes;
	std::vector<cv::Mat> weights;
	std::vector<double> biases;
	int kernelSize;
	int nJob;

	Model(){}; // cannot use no-argument constructor

	// class inside operation function
	bool loadModelFromJSONObject(picojson::object& jsonObj);
	bool loadModelFromBin(std::istream& binFile);
	

	// thread worker function
	bool filterWorker(std::vector<cv::Mat> &inputPlanes,
			std::vector<cv::Mat> &weightMatrices,
			std::vector<cv::Mat> &outputPlanes,
			unsigned int beginningIndex, unsigned int nWorks);

public:
	// ctor and dtor
	Model(picojson::object &jsonObj);
	Model(std::istream& binFile);
	~Model() {}

	// for debugging
	void printWeightMatrix();
	void printBiases();

	// getter function
	int getNInputPlanes();
	int getNOutputPlanes();

	// setter function
	void setNumberOfJobs(int setNJob);

	// public operation function
	bool filter(std::vector<cv::Mat> &inputPlanes,
			std::vector<cv::Mat> &outputPlanes);

	bool filterGL(std::vector<cv::Mat> &inputPlanes,
		std::vector<cv::Mat> &outputPlanes);

	bool saveModelToBin(std::ostream& binFile);
};

class modelUtility {

public:
	static bool generateModelFromJSON(const std::string &fileName,
			std::vector<std::unique_ptr<Model> > &models);
	
	static bool generateModelFromBin(const std::string &fileName,
		std::vector<std::unique_ptr<Model> > &models);
	
	static bool saveModelToBin(const std::string &fileName,
		std::vector<std::unique_ptr<Model> > &models);
};

}

#endif /* MODEL_HANDLER_HPP_ */
