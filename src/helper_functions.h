/*
 * helper_functions.h
 * Some helper functions for the 2D particle filter.
 *  Created on: Dec 13, 2016
 *      Author: Tiffany Huang
 */

#ifndef HELPER_FUNCTIONS_H_
#define HELPER_FUNCTIONS_H_

#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

// for portability of M_PI (Vis Studio, MinGW, etc.)
#ifndef M_PI
const double M_PI = 3.14159265358979323846;
#endif

class Map {
public:

	struct Landmark{

		int id_i ; // Landmark ID
		float x_f; // Landmark x-position in the map (global coordinates)
		float y_f; // Landmark y-position in the map (global coordinates)
	};

	std::vector<Landmark> landmark_list ; // List of landmarks in the map

};

/*
 * Struct representing one landmark observation measurement.
 */
struct LandmarkObs {

	double x;			// Local (vehicle coordinates) x position of landmark observation [m]
	double y;			// Local (vehicle coordinates) y position of landmark observation [m]
};

/**
 * All coordinate arguments are expressed in map coordinates.
 *
 * Output is an index on the vector of landmarks.
 * If a closest neighbor landmark was not found, the output is a negative integer.
 */
inline int closest_neighbor_landmark(
  double pred_x, double pred_y, double sensor_range, const Map &map) {

  double lowest_distance = std::numeric_limits<double>::max();
  int result_ind = -1;

  for (int i = 0; i < map.landmark_list.size(); i++) {
    Map::Landmark lm = map.landmark_list[i];

    double distx = lm.x_f - pred_x;
    double disty = lm.y_f - pred_y;

    // a little optimization
    if (fabs(distx) > sensor_range || fabs(disty) > sensor_range) {
      continue;
    }

    double distance = sqrt(pow(distx, 2) + pow(disty, 2));

    if (distance < lowest_distance) {
      lowest_distance = distance;
      result_ind = i;
    }
  }

  return result_ind;
}

/**
 * Multi-variate Guassian.
 */
inline double observation_error(
  double closest_x, double closest_y,
  double pred_x, double pred_y,
  double stds[]) {

  double std_x = stds[0];
  double std_y = stds[1];

  double exp_term = (pow((closest_x - pred_x) / std_x, 2) + pow((closest_y - pred_y) / std_y, 2)) / -2.0;
  return exp(exp_term) / 2.0 / M_PI / std_x / std_y;
}

/* Reads map data from a file.
 * @param filename Name of file containing map data.
 * @output True if opening and reading file was successful
 */
inline bool read_map_data(std::string filename, Map& map) {

	// Get file of map:
	std::ifstream in_file_map(filename.c_str(),std::ifstream::in);
	// Return if we can't open the file.
	if (!in_file_map) {
		return false;
	}
	
	// Declare single line of map file:
	std::string line_map;

	// Run over each single line:
	while(getline(in_file_map, line_map)){

		std::istringstream iss_map(line_map);

		// Declare landmark values and ID:
		float landmark_x_f, landmark_y_f;
		int id_i;

		// Read data from current line to values::
		iss_map >> landmark_x_f;
		iss_map >> landmark_y_f;
		iss_map >> id_i;

		// Declare single_landmark:
		Map::Landmark single_landmark_temp;

		// Set values
		single_landmark_temp.id_i = id_i;
		single_landmark_temp.x_f  = landmark_x_f;
		single_landmark_temp.y_f  = landmark_y_f;

		// Add to landmark list of map:
		map.landmark_list.push_back(single_landmark_temp);
	}
	return true;
}

#endif /* HELPER_FUNCTIONS_H_ */
