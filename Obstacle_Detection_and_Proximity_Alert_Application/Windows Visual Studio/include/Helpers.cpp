#include "Helpers.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//static vector<float> loadImage(const string& filename, int sizeX = 256, int sizeY = 256)
static vector<float> loadImage(cv::Mat img_frame, int sizeX = 256, int sizeY = 256)
{
    cv::Mat image = img_frame;
    //Convert from BGR to RGB
    cvtColor(image, image, COLOR_BGR2RGB);
    //Resize the image to the input dimension of the OtterNet.
    resize(image, image, Size(sizeX, sizeY));
    //Reshape the image to 1D vector.
    image = image.reshape(1, 1);
    //Normailze number to between 0 and 1 and convert to vector<float>.
    vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    return vec;
}