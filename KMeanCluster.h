// KMeanCluster.cpp
// author: Cheuk-Hang Tse
// This file contains the declaration of the KMeanCluster class.
// This class contains 4 constructors, 1 destructor, and 5 functions
// 
// CONSTRUCTORS:
// KMeanCluster(): define a default clustering model with k equals 1 and train the model based on the default dataset
// KMeanCluster(const int _k): define a default clustering model with k equals the inputted _k and train the model based on the default dataset
// KMeanCluster(const string _fileName): define a default clustering model with k equals 1 and train the model based on the inputted file
// KMeanCluster(const string _fileName, const int _k): define a default clustering model with k equals the inputted _k and train the model based on the inputted file

// DESTRUCTOR:
// ~KMeanCluster(): clear the clusters and points vector

// FUNCTIONS:
// cluster: cluster the inputted point to a cluster and save the new point to the dataset
//			Then, return a vector of fileName that have the same cluster of the inputted points
// readDataSet: read the dataset and converting the entry into Cluster_Point and store them in a vector
// isContainFileName: return true if the fileNames contain _fileName, else false
// saveDataSet: save the clustering points into desire format [filename, point0_x, point0_y, point1_x, ..., pointn_y]
// trainModel: train the k mean cluster model based on the inputted dataset. If dataset is empty, no training is done

#pragma once
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include <algorithm>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class KMeanCluster {
public:
	// KMeanCluster
	// precondition: none
	// postcondition: define a default clustering model with k equals 1 and train the model based on the default dataset
	KMeanCluster();

	// KMeanCluster
	// precondition: _k must be positive
	// postcondition: define a default clustering model with k equals the inputted _k and train the model based on the default dataset
	KMeanCluster(const int _k = 1);

	// KMeanCluster
	// precondition: _fileName must be a valid file
	// postcondition: define a default clustering model with k equals 1 and train the model based on the inputted file
	KMeanCluster(const string _fileName);

	// KMeanCluster
	// precondition: _k must be positive and _fileName must be a valid file
	// postcondition: define a default clustering model with k equals the inputted _k and train the model based on the inputted file
	KMeanCluster(const string _fileName, const int _k = 1);

	// cluster
	// precondition: point must be formatted and inputted correctly, fileN must be the corresponding file of the point
	// postcondition: cluster the inputted point to a cluster and save the new point to the dataset
	//				  Then, return a vector of fileName that have the same cluster of the inputted points
	vector<string> cluster(const vector<double>& point, const string fileName);

	// ~KMeanCluster
	// precondition: none
	// postcondition: clear the clusters and points vector
	~KMeanCluster();

private:

	// isContainFileName
	// precondition: fileNames is not empty, _fileName must be a valid string
	// postcondition: return true if the fileNames contain _fileName, else false
	bool isContainFileName(const vector<string> fileNames, const string _fileName);

	// readDataSet
	// precondition: _fileName should be a valid file name
	// postcondition: read the dataset and converting the entry into Cluster_Point and store them in a vector
	void readDataSet(const string _fileName);

	// saveDataSetsa
	// precondition: _fileName must be a valid file name
	// postcondition: save the clustering points into desire format [filename, point0_x, point0_y, point1_x, ..., pointn_y]
	void saveDataSet(const string _fileName);

	// trainModel
	// precondition: none
	// postcondition: train the k mean cluster model based on the inputted dataset. If dataset is empty, no training is done
	void trainModel();

	// Cluster_Point
	// A structure that store information of an image human pose points
	struct Cluster_Point {
		vector<double> coords;
		int clusterId;
		double minDistance;
		string fileName;

		// Default constructor
		Cluster_Point() {
			clusterId = -1;
			minDistance = std::numeric_limits<double>::max();
		}

		// set the Cluster_Point to inputted coordinates and name of the image file
		Cluster_Point(const vector<double>& point, const string name) {
			coords = point;
			clusterId = -1;
			fileName = name;
			minDistance = std::numeric_limits<double>::max();
		}

		// distance
		// precondition: p is not empty
		// postcondition: return the distance between two cluster points
		double distance(Cluster_Point p) {
			double sum = 0;
			if (p.coords.size() != coords.size())
				return -1.0;
			int i = 0;
			while (i < p.coords.size()) {
				sum += pow((p.coords[i] - coords[i]), 2);
				i += 1;
			}
			return sqrt(sum);
		}
	};

	vector<Cluster_Point> points; // all the Cluster_Points in the dataset
	vector<Cluster_Point> clusters; // all the centroid of the K-Mean clustering
	string fileName; // dataset file name
	int k; // number of k
	int trainIteration = 100;
};