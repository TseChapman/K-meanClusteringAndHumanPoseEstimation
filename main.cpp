// main.cpp
// author: Cheuk-Hang Tse
// The code includes 3 functions: validateParameters, pre_processPoints, and showRelatedPoseImages
// validateParameters: Return true if the device is "gpu" or "cpu", else false
// pre_processPoints: convert the Points vector into a normalized double vector
// showRelatedPoseImages: show all the image based on the file names within the fileNames vector
// This functions are used to perform human pose estimation and find similar images
// Author: Cheuk-Hang Tse

#include "HumanPoseEstimation.h"
#include "KMeanCluster.h"

// validateParameters
// precondition: device is inputted correctly
// postcondition: returns a boolean. True if device is either "gpu"/"cpu". Else, false
bool validateParameters(const string device) {
    return (device == "gpu" || device == "cpu");
}

// pre_processPoints
// precondition: vector of points should not be empty
// postcondition: convert the Points vector into a normalized double vector
vector<double> pre_processPoints(const vector<Point>& v) {
	// Convert points into a Cluster_Point
	vector<double> p;
	double maxX = 0;
	double minX = std::numeric_limits<double>::max();
	double maxY = 0;
	double minY = std::numeric_limits<double>::max();
	for (int i = 0; i < v.size(); i++) {
		maxX = max(maxX, (double)v.at(i).x);
		maxY = max(maxY, (double)v.at(i).y);
		minX = min(minX, (double)v.at(i).x);
		minY = min(minY, (double)v.at(i).y);
	}
	for (int i = 0; i < v.size(); i++) {
		p.push_back(((double)v.at(i).x - minX) / (maxX - minX));
		p.push_back(((double)v.at(i).y - minY) / (maxY - minY));
	}
	return p;
}

// showRelatedPoseImages
// precondition: fileNames is not an empty string vector
// postcondition: show all the image based on the file names within the fileNames vector
void showRelatedPoseImages(const vector<string> fileNames) {
	for (int i = 0; i < fileNames.size(); i++) {
		Mat frame = imread(fileNames.at(i));
		string name = "realated-image" + to_string(i);
		imshow(name, frame);
	}
	waitKey();
}

// main
// precondition: there must be 3 parameters: device (gpu/cpu), inputFile (file name with opencv readable file type), k (number of clusters in k mean)
// postconditions: Use input parameters to get input image file and perform human pose estimation using a Multi-Person Dataset (MPII) deep neutral network model
//					The model will produce at most 15 joint pixel locations. These points will be displayed in a window
//					Next, use the point locations to run a k-mean clustering and find similar images
//					All the similar images are displayed
int main(int argc, char* argv[])
{
    // Code must have 3 parameters
    if (argc != 4)
        return -1;

    // Read parameters
    string device = argv[1];
    string inputFile = argv[2];
	int k = stoi(argv[3]);

    // Validate parameters
    if (!validateParameters(device))
        return -1;

	cout << "Start Human Pose Estimation using " << device << " on file " << inputFile << endl;

	int inWidth = 368;
	int inHeight = 368;
	float thresh = 0.1;
	vector<Point> v =performHumanPoseEstimation(device, inputFile, inWidth, inHeight, thresh);
	// Convert points into a double
	vector<double> p = pre_processPoints(v);
	// Compute Clustering
	KMeanCluster kCluster(k);
	vector<string> files = kCluster.cluster(p, inputFile);
	std::ofstream out("test.txt");

	for (const auto& row : files) {
		out << row << '\n';
	}
	out.close();

	showRelatedPoseImages(files);
    return 0;
}