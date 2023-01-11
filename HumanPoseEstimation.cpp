// HumanPoseEstimation.cpp
// author: Cheuk-Hang Tse
// This file has 3 functions: findBodyPartPosition, drawPointsConnection, and performHumanPoseEstimation
// findBodyPartPosition: Return the point locations in a form of a vector
// drawPointsConnection: draw points and make a directly straight line connection between the point pair in the inputted frame
// performHumanPoseEstimation: use deep neural network to find point locations and display the human pose
// Source: https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

#include "HumanPoseEstimation.h"

// MPI
// define the parameters of the deep neural network
#ifdef MPI
const int POSE_PAIRS[14][2] =
{
	{0,1}, {1,2}, {2,3},
	{3,4}, {1,5}, {5,6},
	{6,7}, {1,14}, {14,8}, {8,9},
	{9,10}, {14,11}, {11,12}, {12,13}
};

string prototxt = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
string weightsModel = "pose/mpi/pose_iter_160000.caffemodel";

int nPoints = 15;
#endif

// findBodyPartPosition
// precondition: output and frameCopy is not empty, and other parameters are inputed correctly
// postcondition: Return the point locations in a form of a vector
vector<Point> findBodyPartPosition(Mat& output, const float thresh, const int frameWidth, const int frameHeight, const Mat& frameCopy) {
    int H = output.size[2];
    int W = output.size[3];

    // find the position of the body parts
    vector<Point> points(nPoints);
    for (int n = 0; n < nPoints; n++)
    {
        // Probability map of corresponding body's part.
        Mat probMap(H, W, CV_32F, output.ptr(0, n));

        Point2f p(-1, -1);
        Point maxLoc;
        double prob;
        minMaxLoc(probMap, 0, &prob, 0, &maxLoc);

        // Check if the prob is above the inputted threshold
        if (prob > thresh)
        {
            p = maxLoc;
            p.x *= (float)frameWidth / W;
            p.y *= (float)frameHeight / H;

            circle(frameCopy, cv::Point((int)p.x, (int)p.y), 8, Scalar(0, 255, 255), -1);
            cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        }
        points[n] = p;
    }
    cout << points[0] << endl;
    return points;
}

// drawpointsConnection
// preconditions: frame is not an empty image
// postcondition: draw points and make a directly straight line connection between the point pair in the inputted frame
void drawPointsConnection(const int nPairs, const vector<Point>& points, const Mat& frame) {
    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected points pair
        Point2f partA = points[POSE_PAIRS[n][0]];
        Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
            continue;

        line(frame, partA, partB, Scalar(0, 255, 255), 8);
        circle(frame, partA, 8, Scalar(0, 0, 255), -1);
        circle(frame, partB, 8, Scalar(0, 0, 255), -1);
    }
}

// performHumanPoseEstimation
// preconditions: input parameters are inputted correctly and not empty
// postconditions: use deep neural network to find point locations and display the human pose
vector<Point> performHumanPoseEstimation(const string device, const string imageFile, const int inWidth, const int inHeight, const float thresh) {
    // Read the image file
    Mat frame = imread(imageFile);

    // Check if the file is empty
    if (frame.empty())
    {
        std::cout << "Could not read the image: " << imageFile << std::endl;
        exit(-1);
    }

    // Make a clone from the image and collect image pixels length and width
    Mat frameCopy = frame.clone();
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;

    // Get the dnn model from caffe
    double t = (double)cv::getTickCount();
    Net netModel = readNetFromCaffe(prototxt, weightsModel);

    // Set which device is used for the model
    if (device == "cpu")
    {
        netModel.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device == "gpu")
    {
        netModel.setPreferableBackend(DNN_BACKEND_CUDA);
        netModel.setPreferableTarget(DNN_TARGET_CUDA);
    }

    // format the image for the network
    Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

    netModel.setInput(inpBlob);

    // get the processed image from the dnn model
    Mat output = netModel.forward();

    // Find the points based on a threshold
    vector<Point> points = findBodyPartPosition(output, thresh, frameWidth, frameHeight, frameCopy);

    // Draw the pose estimation and display the image
    int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]); 
    drawPointsConnection(nPairs, points, frame);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Time Taken = " << t << endl;
    imshow("Output-Keypoints", frameCopy);
    imshow("Output-Skeleton", frame);
    imwrite("Output-Skeleton.jpg", frame);
    waitKey();
    return points;
}