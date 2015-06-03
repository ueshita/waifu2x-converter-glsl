
#include <exception>
#include "convertRoutine.hpp"
#include "filterGL.h"

namespace w2xc {

// converting process inside program
static bool convertWithModelsBasic(cv::Mat &inputPlane, cv::Mat &outputPlane,
		std::vector<std::unique_ptr<Model> > &models);
static bool convertWithModelsBlockSplit(cv::Mat &inputPlane,
		cv::Mat &outputPlane, std::vector<std::unique_ptr<Model> > &models);

bool convertWithModels(cv::Mat &inputPlane, cv::Mat &outputPlane,
		std::vector<std::unique_ptr<Model> > &models, bool blockSplitting) {

	cv::Size blockSize = modelUtility::getInstance().getBlockSize();
	bool requireSplitting = (inputPlane.size().width * inputPlane.size().height)
			> blockSize.width * blockSize.height;
//	requireSplitting = true;
	if (blockSplitting && requireSplitting) {
		return convertWithModelsBlockSplit(inputPlane, outputPlane, models);
	} else {
		//insert padding to inputPlane
		cv::Mat tempMat;
		int nModel = models.size();
		cv::Size outputSize = inputPlane.size();
		cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel,
				cv::BORDER_REPLICATE);

		bool ret = convertWithModelsBasic(tempMat, outputPlane, models);
		if (ret == false) {
			return false;
		}

		tempMat = outputPlane(cv::Range(nModel, outputSize.height + nModel),
				cv::Range(nModel, outputSize.width + nModel));
		assert(
				tempMat.size().width == outputSize.width
						&& tempMat.size().height == outputSize.height);

		tempMat.copyTo(outputPlane);

		return ret;
	}

}

static bool convertWithModelsBasic(cv::Mat &inputPlane, cv::Mat &outputPlane,
		std::vector<std::unique_ptr<Model> > &models) {

	cv::Size size = inputPlane.size();

	try {
		// initialize GL filter core
		filterGLInit(size.width, size.height);

		// setup shader per model
		for (int index = 0; index < (int)models.size(); index++) {
			if (!models[index]->loadGLShader()) {
				std::exit(-1);
			}
		}

		// set the input image data
		cv::Mat tempPlane = cv::Mat::zeros(size, CV_32FC1);
		inputPlane.copyTo(tempPlane);
		filterGLSetInputData(tempPlane);

		for (int index = 0; index <= (int)models.size(); index++) {
			
			//std::cout << "Iteration #" << (index + 1) << "..." << std::endl;
			
			std::cout << "\r[";
			int progress = 0;
			for (; progress < index; progress++)               std::cout << "=";
			for (; progress < (int)models.size(); progress++)  std::cout << " ";
			std::cout << "]";
			std::cout.flush();

			if (index >= (int)models.size()) {
				break;
			}
			
			// core processing
			if (!models[index]->filterGL(index)) {
				std::exit(-1);
			}
		}
		// get the output image data
		filterGLGetOutputData(tempPlane);
		tempPlane.copyTo(outputPlane);
		
		std::cout << " ok" << std::endl;
		
		// finalize GL filter core
		filterGLRelease();
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}

	return true;

}

static bool convertWithModelsBlockSplit(cv::Mat &inputPlane,
		cv::Mat &outputPlane, std::vector<std::unique_ptr<Model> > &models) {

	// padding is not required before calling this function

	// initialize local variables
	cv::Size blockSize = modelUtility::getInstance().getBlockSize();

	unsigned int nModel = models.size();

	//insert padding to inputPlane
	cv::Mat tempMat;
	cv::Size outputSize = inputPlane.size();
	cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel,
			cv::BORDER_REPLICATE);

	// calcurate split rows/cols
	unsigned int splitColumns = static_cast<unsigned int>(std::ceil(
			static_cast<float>(outputSize.width)
					/ static_cast<float>(blockSize.width - 2 * nModel)));
	unsigned int splitRows = static_cast<unsigned int>(std::ceil(
			static_cast<float>(outputSize.height)
					/ static_cast<float>(blockSize.height - 2 * nModel)));
	
	std::cout << "split blocks " << splitRows << "x" << splitColumns << " ..."
			  << std::endl;

	// start to convert
	cv::Mat processRow;
	cv::Mat processBlock;
	cv::Mat processBlockOutput;
	cv::Mat writeMatTo;
	cv::Mat writeMatFrom;
	outputPlane = cv::Mat::zeros(outputSize, CV_32FC1);
	for (unsigned int r = 0; r < splitRows; r++) {
		if (r == splitRows - 1) {
			processRow = tempMat.rowRange(r * (blockSize.height - 2 * nModel),
					tempMat.size().height);
		} else {
			processRow = tempMat.rowRange(r * (blockSize.height - 2 * nModel),
					r * (blockSize.height - 2 * nModel) + blockSize.height);
		}
		for (unsigned int c = 0; c < splitColumns; c++) {
			if (c == splitColumns - 1) {
				processBlock = processRow.colRange(
						c * (blockSize.width - 2 * nModel),
						tempMat.size().width);
			} else {
				processBlock = processRow.colRange(
						c * (blockSize.width - 2 * nModel),
						c * (blockSize.width - 2 * nModel) + blockSize.width);
			}

			std::cout << "process block (" << (c + 1) << "," << (r + 1) << ") ..."
					<< std::endl;
			if (!convertWithModelsBasic(processBlock, processBlockOutput,
					models)) {
				std::cerr << "w2xc::convertWithModelsBasic()\n"
						"in w2xc::convertWithModelsBlockSplit() : \n"
						"something error has occured. stop." << std::endl;
				return false;
			}

			writeMatFrom = processBlockOutput(
					cv::Range(nModel,
							processBlockOutput.size().height - nModel),
					cv::Range(nModel,
							processBlockOutput.size().width - nModel));
			writeMatTo = outputPlane(
					cv::Range(r * (blockSize.height - 2 * nModel),
							r * (blockSize.height - 2 * nModel)
									+ processBlockOutput.size().height
									- 2 * nModel),
					cv::Range(c * (blockSize.height - 2 * nModel),
							c * (blockSize.height - 2 * nModel)
									+ processBlockOutput.size().width
									- 2 * nModel));
			assert(
					writeMatTo.size().height == writeMatFrom.size().height
							&& writeMatTo.size().width
									== writeMatFrom.size().width);
			writeMatFrom.copyTo(writeMatTo);

		} // end process 1 column

	} // end process all blocks

	return true;

}

}

