// HumanPoseEstimation.h
// author: Cheuk-Hang Tse
// This file has 3 functions: findBodyPartPosition, drawPointsConnection, and performHumanPoseEstimation
// These functions allow human pose estimation on an image and return the skeleton of the human pose within the image
// findBodyPartPosition: Return the point locations in a form of a vector
// drawPointsConnection: draw points and make a directly straight line connection between the point pair in the inputted frame
// performHumanPoseEstimation: use deep neural network to find point locations and display the human pose
// Source: https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

#pragma once

#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <thread>

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

// MPI
// define the parameters of the deep neural network
#define MPI

// findBodyPartPosition
// precondition: output and frameCopy is not empty, and other parameters are inputed correctly
// postcondition: Return the point locations in a form of a vector
vector<Point> findBodyPartPosition(Mat& output, const float thresh, const int frameWidth, const int frameHeight, const Mat& frameCopy);

// drawpointsConnection
// preconditions: frame is not an empty image
// postcondition: draw points and make a directly straight line connection between the point pair in the inputted frame
void drawPointsConnection(const int nPairs, const vector<Point>& points, const Mat& frame);

// performHumanPoseEstimation
// preconditions: input parameters are inputted correctly and not empty
// postconditions: use deep neural network to find point locations and display the human poses
vector<Point> performHumanPoseEstimation(const string device, const string imageFile, const int inWidth, const int inHeight, const float thresh);