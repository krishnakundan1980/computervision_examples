#include "StdAfx.h"
#include "ImageHazeEstimator.h"


ImageHazeEstimator::ImageHazeEstimator(void)
{
}


ImageHazeEstimator::~ImageHazeEstimator(void)
{
}

long double ImageHazeEstimator::getHzeScore(const Mat3b& haze_input_img )
{
	Mat3b input_img_in_double;
	haze_input_img.convertTo( input_img_in_double, CV_8UC3 );
		
	Mat3b input_img;
	cv::resize(input_img_in_double, input_img, cv::Size(), 0.10, 0.10);
				
	Mat1b dark, bright;
	getDarkAndBrightColorChannel(input_img, dark, bright);

	Mat dd;
	cv::reduce(dark, dd, 1, CV_REDUCE_SUM, CV_32F);
	dd = dd/(input_img.cols*input_img.rows);

	Mat bb;
	cv::reduce(bright, bb, 1, CV_REDUCE_SUM, CV_32F);
	bb = bb/(input_img.cols*input_img.rows);

	Mat cc;
	cv::absdiff(dd,bb,cc);						

	Mat bright_max;
	cv::reduce(bright, bright_max, 1, CV_REDUCE_MAX);	
	bright_max = bright_max/3;

		
	Mat bb_tmp;
	bb_tmp = (2*bb)/3;
		
	Mat atmosphere;
	cv::add(bright_max, bb_tmp, atmosphere,Mat(),CV_32F);
		
	Mat X1_tmp;
	cv:subtract(atmosphere, dd,X1_tmp);			

	Mat X1 = Mat::zeros(dd.rows,dd.cols,CV_32F);		
	X1 = mldivide(atmosphere, X1_tmp);
	float X11 = X1.at<float>(0, 0);

	Mat X2= Mat::zeros(dd.rows,dd.cols,CV_32F);
	X2 = mldivide(atmosphere,cc);
	float X22 = X2.at<float>(0, 0);		

	long double mu = 5.1f;
	long double nu = 2.9f;
	long double sigma = .2461f;
	long double haze_metric = cv::exp((sigma - (mu*X11+nu*X22)/2));

	return haze_metric;
}

void ImageHazeEstimator::getDarkAndBrightColorChannel(const Mat3b& img, Mat1b& dark, Mat1b& bright)
{
	int win_size = 15;
    DarkChannel(img, dark, 15);
    BrightChannel(img, bright, 15);
}

void ImageHazeEstimator::DarkChannel(const Mat3b& img, Mat1b& dark, int patchSize)
{
    int radius = patchSize / 2;
    Mat1b low;
    minValue3b(img, low);
    minFilter(low, dark, radius);
}

void ImageHazeEstimator::BrightChannel(const Mat3b& img, Mat1b& bright, int patchSize)
{
    int radius = patchSize / 2;
    Mat1b high;
    maxValue3bBrightChannel(img, high);
    maxFilterBrightChannel(high, bright, radius);
}

cv::Mat ImageHazeEstimator::mldivide(const cv::Mat& A, const cv::Mat& B ) 
{
    cv::Mat a;
    cv::Mat b;
    A.convertTo( a, CV_64FC3 );
    B.convertTo( b, CV_64FC3 );

    cv::Mat ret;
    cv::solve( a, b, ret, cv::DECOMP_NORMAL );

    cv::Mat ret2;
    ret.convertTo( ret2, A.type() );
    return ret2;
}

void ImageHazeEstimator::minValue3b(const Mat3b& src, Mat1b& dst)
{
    int rr = src.rows;
    int cc = src.cols;

    dst = Mat1b(rr, cc, uchar(0));

    for (int c = 0; c<cc; ++c)
    {
        for (int r = 0; r<rr; ++r)
        {
            const Vec3b& v = src(r, c);

            uchar lowest = v[0];
            if (v[1] < lowest) lowest = v[1];
            if (v[2] < lowest) lowest = v[2];
            dst(r, c) = lowest;
        }
    }
}

void ImageHazeEstimator::minFilter(const Mat1b& src, Mat1b& dst, int radius)
{
    Mat1b padded;
    copyMakeBorder(src, padded, radius, radius, radius, radius, BORDER_CONSTANT, Scalar(255));

    int rr = src.rows;
    int cc = src.cols;
    dst = Mat1b(rr, cc, uchar(0));

    for (int c = 0; c < cc; ++c)
    {
        for (int r = 0; r < rr; ++r)
        {
            uchar lowest = 255;
            for (int i = -radius; i <= radius; ++i)
            {
                for (int j = -radius; j <= radius; ++j)
                {
                    uchar val = padded(radius + r + i, radius + c + j);
                    if (val < lowest) lowest = val;
                }
            }
            dst(r, c) = lowest;
        }
    }
}

void ImageHazeEstimator::maxFilterBrightChannel(const Mat1b& src, Mat1b& dst, int radius)
{
    Mat1b padded;
    copyMakeBorder(src, padded, radius, radius, radius, radius, BORDER_CONSTANT, Scalar(255));

    int rr = src.rows;
    int cc = src.cols;
    dst = Mat1b(rr, cc, uchar(0));

    for (int c = 0; c < cc; ++c)
    {
        for (int r = 0; r < rr; ++r)
        {
            uchar lowest = 0;
            for (int i = -radius; i <= radius; ++i)
            {
                for (int j = -radius; j <= radius; ++j)
                {
                    uchar val = padded(radius + r + i, radius + c + j);
                    if (val > lowest) lowest = val;
                }
            }
            dst(r, c) = lowest;
        }
    }
}

void ImageHazeEstimator::maxValue3bBrightChannel(const Mat3b& src, Mat1b& dst)
{
    int rr = src.rows;
    int cc = src.cols;

    dst = Mat1b(rr, cc, uchar(0));

    for (int c = 0; c<cc; ++c)
    {
        for (int r = 0; r<rr; ++r)
        {
            const Vec3b& v = src(r, c);

            uchar lowest = v[0];
            if (v[1] > lowest) lowest = v[1];
            if (v[2] > lowest) lowest = v[2];
            dst(r, c) = lowest;
        }
    }
}