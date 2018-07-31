// Test2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "ImageHazeEstimator.h"
using namespace std;
using namespace cv;

String type2str(int type);

static void help()
{
    cout<< "------------------------------------------------------------------------------" << endl
        << "This program shows how to get haze estimate for an image using OpenCV. In addition, it "
        << "keeps the algorithm into a separate class definition to separate the logic from main program."       
        << "--------------------------------------------------------------------------"     << endl
        << endl;
}

//string ty =  type2str( frameReference.type() );
//printf("Matrix: %s %dx%d \n", ty.c_str(), frameReference.cols, frameReference.rows );

int _tmain(int argc, _TCHAR* argv[])
{
	help();

    if (argc != 2)
    {
        cout << "Not enough parameters.. Please enter a path of video file for which haze estimation is required!" << endl;
        return -1;
    }

    //stringstream conv;    
    //conv << argv[1] << endl ;       // put in the strings    

	const string sourceReference = argv[1];    
    int frameNum = -1;          // Frame counter
	int delay = 10;

    VideoCapture captRefrnc(sourceReference);
    if (!captRefrnc.isOpened())
    {
        cout  << "Could not open video file.." << sourceReference << endl;
        return -1;
    }
 
	//Get input video file size
    Size refS = Size((int) captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
                     (int) captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT));
 
	const char* WIN_RF = "Reference";

    namedWindow(WIN_RF, WINDOW_AUTOSIZE);    
    moveWindow(WIN_RF, 400 , 0);     
    cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
         << " of nr#: " << captRefrnc.get(CV_CAP_PROP_FRAME_COUNT) << endl;
  
    Mat3b frameReference;   
    while(true) //Show the image captured in the window and loop through for the next image sequence, until video reaches to the last frame
    {
        captRefrnc >> frameReference;
        if (frameReference.empty())
        {
            cout << " < < <  No frame left to process!  > > > ";
            break;
        }
        ++frameNum;

        cout << "Frame: " << frameNum << "# ";
        				
		if(frameNum == 30)//Only process 1 frame every second to minimize the CPU usage
		{
			ImageHazeEstimator hazeEstimator;
			long double haze_metric = hazeEstimator.getHzeScore(frameReference);				
			String s1 = std::to_string(haze_metric);			

			if(haze_metric > 1.4)//Haze thresold value to mark haze/non-haze
			{
				putText(frameReference, s1+"Smoke", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

				/*Mat watermark = imread("HsmokeWatermark.jpg");
				watermarked = (0.8 * frameReference) +  (0.2 * watermark);*/
			}
			else
			{
				putText(frameReference, s1+"No Smoke", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);				
			}
			frameNum = 0;
		}

		//Show
        imshow(WIN_RF, frameReference);

        char c = (char)waitKey(delay);
        if (c == 27) break;
    }

    return 0;
}

//Method to determine video color space storage parameters
String type2str(int type) {
	String r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}