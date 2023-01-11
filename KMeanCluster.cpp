// KMeanCluster.cpp
// author: Cheuk-Hang Tse
// This file contains the implementation of the KMeanCluster class.
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


#include "KMeanCluster.h"

// KMeanCluster
// precondition: none
// postcondition: define a default clustering model with k equals 1 and train the model based on the default dataset
KMeanCluster::KMeanCluster() {
	fileName = "test.csv";
	k = 1;
	readDataSet(fileName);
	trainModel();
}

// KMeanCluster
// precondition: _k must be positive
// postcondition: define a default clustering model with k equals the inputted _k and train the model based on the default dataset
KMeanCluster::KMeanCluster(const int _k) {
	fileName = "test.csv";
	k = _k;
	readDataSet(fileName);
	trainModel();
}

// KMeanCluster
// precondition: _fileName must be a valid file
// postcondition: define a default clustering model with k equals 1 and train the model based on the inputted file
KMeanCluster::KMeanCluster(const string _fileName) {
	fileName = _fileName;
	k = 1;
	readDataSet(fileName);
	trainModel();
}

// KMeanCluster
// precondition: _k must be positive and _fileName must be a valid file
// postcondition: define a default clustering model with k equals the inputted _k and train the model based on the inputted file
KMeanCluster::KMeanCluster(const string _fileName, const int _k) {
	fileName = _fileName;
	k = _k;
	readDataSet(fileName);
	trainModel();
}

// cluster
// precondition: point must be formatted and inputted correctly, fileN must be the corresponding file of the point
// postcondition: cluster the inputted point to a cluster and save the new point to the dataset
//				  Then, return a vector of fileName that have the same cluster of the inputted points
vector<string> KMeanCluster::cluster(const vector<double>& point, const string fileN) {
	Cluster_Point cp(point, fileN);


	// Determine if the dataset is empty
	if (!point.size()) {
		points.push_back(cp);
		saveDataSet(fileName);
		return vector<string>();
	}

	// Find which centroid it belongs to
	for (int i = 0; i < clusters.size(); i++) {
		int clusterId = i;
		double dist = clusters.at(i).distance(cp);
		if (dist < cp.minDistance) {
			cp.minDistance = dist;
			cp.clusterId = clusterId;
		}
	}

	// Find points's fileName
	int clusterTarget = cp.clusterId;
	vector<string> fileNames;
	for (int i = 0; i < points.size(); i++) {
		if (points.at(i).clusterId == clusterTarget) {
			fileNames.push_back(points.at(i).fileName);
		}
	}
	points.push_back(cp);
	saveDataSet(fileName);
	return fileNames;
}

// ~KMeanCluster
// precondition: none
// postcondition: clear the clusters and points vector
KMeanCluster::~KMeanCluster() {
	clusters.clear();
	points.clear();
}

// isContainFileName
// precondition: fileNames is not empty, _fileName must be a valid string
// postcondition: return true if the fileNames contain _fileName, else false
bool KMeanCluster::isContainFileName(const vector<string> fileNames, const string _fileName) {
	if (std::find(fileNames.begin(), fileNames.end(), _fileName) != fileNames.end()) {
		return true;
	}
	else {
		return false;
	}

}

// readDataSet
// precondition: _fileName should be a valid file name
// postcondition: read the dataset and converting the entry into Cluster_Point and store them in a vector
void KMeanCluster::readDataSet(const string _fileName) {
	try {
		points.clear();
		string line, word;

		fstream file(_fileName, ios::in);
		if (file.is_open())
		{
			while (getline(file, line))
			{
				Cluster_Point p;

				stringstream str(line);
				int i = 0;
				while (getline(str, word, ',')) {
					if (i == 0)
					{
						p.fileName = word;
					}
					else {
						p.coords.push_back(stod(word));
					}
					i++;
				}
				points.push_back(p);
			}
		}
		else
			cout << "Could not open the file\n";
	}
	catch (exception e) {
		cerr << e.what();
		exit(-1);
	}
}

// saveDataSet
// precondition: _fileName must be a valid file name
// postcondition: save the clustering points into desire format [filename, point0_x, point0_y, point1_x, ..., pointn_y]
void KMeanCluster::saveDataSet(const string _fileName) {
	std::ofstream out(_fileName);

	vector<string> fileNames;
	for (auto& row : points) {
		if (isContainFileName(fileNames, row.fileName))
			continue;

		out << row.fileName << ',';
		fileNames.push_back(row.fileName);
		for (auto col : row.coords)
			out << col << ',';
		out << '\n';
	}
}

// trainModel
// precondition: none
// postcondition: train the k mean cluster model based on the inputted dataset. If dataset is empty, no training is done
void KMeanCluster::trainModel() {
	try {
		// Determine if the dataset is empty
		if (!points.size())
			return;

		// Initialize k cluster points
		srand((unsigned int)time(0));
		size_t n = points.size();
		for (int i = 0; i < k; i++) {
			clusters.push_back(points.at(rand() % n));
		}

		// Iterate x times of the steps below
		for (int l = 0; l < trainIteration; l++) {
			// Assign points to a cluster
			for (int i = 0; i < clusters.size(); i++) {
				int clusterId = i;
				for (int j = 0; j < points.size(); j++) {
					double dist = clusters.at(i).distance(points.at(j));
					if (dist < points.at(j).minDistance) {
						points.at(j).minDistance = dist;
						points.at(j).clusterId = clusterId;
					}
				}
			}
			// Recompute Centroids
			vector<int> nPoints;
			vector<vector<double>> sumCoord;
			for (int i = 0; i < clusters.size(); i++) {
				nPoints.push_back(0);
				vector<double> v(clusters.at(i).coords.size(), 0);
				sumCoord.push_back(v);
			}
			for (int i = 0; i < points.size(); i++) {
				int clusterId = points.at(i).clusterId;
				nPoints.at(clusterId) += 1;
				for (int j = 0; j < points.at(i).coords.size(); j++) {
					sumCoord.at(clusterId).at(j) += points.at(i).coords.at(j);
				}
				points.at(i).minDistance = std::numeric_limits<double>::max();
			}

			for (int i = 0; i < clusters.size(); i++) {
				int clusterId = clusters.at(i).clusterId;
				for (int j = 0; j < clusters.at(i).coords.size(); j++) {
					clusters.at(i).coords[j] = sumCoord.at(clusterId).at(j) / nPoints.at(clusterId);
				}
			}
		}
		
	}
	catch (exception e) {
		cerr << e.what();
	}
}