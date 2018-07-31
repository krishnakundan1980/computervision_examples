#pragma once

#include "stdafx.h"

using namespace std;
using namespace cv;

class ImageHazeEstimator
{
	cv::Mat ImageHazeEstimator::mldivide(const cv::Mat& A, const cv::Mat& B );
	void getDarkAndBrightColorChannel(const Mat3b& img, Mat1b& dark, Mat1b& bright);
	void DarkChannel(const Mat3b& img, Mat1b& dark, int patchSize);
	void BrightChannel(const Mat3b& img, Mat1b& bright, int patchSize);
	void minValue3b(const Mat3b& src, Mat1b& dst);
	void ImageHazeEstimator::minFilter(const Mat1b& src, Mat1b& dst, int radius);
	void maxFilterBrightChannel(const Mat1b& src, Mat1b& dst, int radius);
	void maxValue3bBrightChannel(const Mat3b& src, Mat1b& dst);

public:
	ImageHazeEstimator(void);
	~ImageHazeEstimator(void);
	long double getHzeScore(const Mat3b& img );
};
