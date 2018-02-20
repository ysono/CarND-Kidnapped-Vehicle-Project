/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <random>

#include "helper_functions.h"

struct Particle {

	double x;
	double y;
	double theta;

	// associations: The landmark id that goes along with each association
	std::vector<int> associations;

	// sense_x: the associations x mapping already converted to world coordinates
	std::vector<double> sense_x;

	// sense_y: the associations y mapping already converted to world coordinates
	std::vector<double> sense_y;
};

class ParticleFilter {
	
	// Flag, if filter is initialized
	bool is_initialized;

	// Number of particles to draw
	int num_particles;

	// Note, if standard deviations are assumed constant, we can also instantiate
	// std::normal_distribution objects as class memberes.
	// STDs are constant in the simulation, but in real life they should differ for
	// every measurement, so this optimization is skippeed.
	std::mt19937 gen;

	// Vector of weights of all particles
	std::vector<double> weights;

public:
	
	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param num_particles Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {
		std::random_device rd;
		gen = std::mt19937(rd());
	}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double gps_x, double gps_y, double gps_theta, double gps_std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/**
	 * Get the Particle with the largest weight.
	 */
	Particle bestParticle();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle setAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y);
	
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized;
	}
};

#endif /* PARTICLE_FILTER_H_ */
