/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double gps_std[]) {

  num_particles = 8;

  std::normal_distribution<double> dist_x(gps_x, gps_std[0]);
  std::normal_distribution<double> dist_y(gps_y, gps_std[1]);
  std::normal_distribution<double> dist_theta(gps_theta, gps_std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    particles.push_back(p);

    weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for (std::vector<Particle>::iterator iter = particles.begin(); iter != particles.end(); iter++) {
    Particle p = *iter;

    double delta_x, delta_y, delta_theta;
    if (fabs(yaw_rate) > 0.001) {
      double c0 = velocity / yaw_rate;
      double delta_yaw = yaw_rate * delta_t;
      delta_x = c0 * (sin(p.theta + delta_yaw) - sin(p.theta));
      delta_y = c0 * (cos(p.theta) - cos(p.theta + delta_yaw));
      delta_theta = delta_yaw;
    } else {
      delta_x = velocity * delta_t * cos(p.theta);
      delta_y = velocity * delta_t * sin(p.theta);
      delta_theta = 0;
    }

    p.x += delta_x + dist_x(gen);
    p.y += delta_y + dist_y(gen);
    p.theta += delta_theta + dist_theta(gen);

    *iter = p;
  }
}

void ParticleFilter::updateWeights(
    double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map) {

  weights.clear();

  for (std::vector<Particle>::iterator iter = particles.begin(); iter != particles.end(); iter++) {

    Particle p = *iter;

    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    double cos_p_theta = cos(p.theta);
    double sin_p_theta = sin(p.theta);

    double weight = 1.0;

    for (int obs_ind = 0; obs_ind < observations.size(); obs_ind++) {
      LandmarkObs obs = observations[obs_ind];

      double predicted_landmark_x = cos_p_theta * obs.x - sin_p_theta * obs.y + p.x;
      double predicted_landmark_y = sin_p_theta * obs.x + cos_p_theta * obs.y + p.y;

      int neighbor_landmark_ind = closest_neighbor_landmark(
        predicted_landmark_x, predicted_landmark_y, sensor_range, map);

      if (neighbor_landmark_ind < 0) {
        // It may be bettter to assign a fraction of the total weight which is calculated later.
        // In any case empirically this `if` condition is never encountered.
        weight = 1e-9;
        break;
      }

      Map::Landmark neighbor_landmark = map.landmark_list[neighbor_landmark_ind];

      double closest_x = neighbor_landmark.x_f;
      double closest_y = neighbor_landmark.y_f;

      p.associations.push_back(neighbor_landmark.id_i);
      p.sense_x.push_back(predicted_landmark_x);
      p.sense_y.push_back(predicted_landmark_y);

      double prob_obs = observation_error(
        closest_x, closest_y,
        predicted_landmark_x, predicted_landmark_y,
        std_landmark);
      weight *= prob_obs;
    }

    *iter = p;
    
    weights.push_back(weight);
  }
}

void ParticleFilter::resample() {

  std::discrete_distribution<int> dist_weights(weights.begin(), weights.end());

  std::vector<Particle> resampled(num_particles);

  for (int i = 0; i < num_particles; i++) {
    int selected_index = dist_weights(gen);
    resampled[i] = particles[selected_index];
  }

  particles = resampled;
}

Particle ParticleFilter::bestParticle() {

  double highest_weight = -1.0;
  int highest_weight_index;
  for (int weight_ind = 0; weight_ind < num_particles; weight_ind++) {
    double w = weights[weight_ind];
    if (w > highest_weight) {
      highest_weight = w;
      highest_weight_index = weight_ind;
    }
  }
  return particles[highest_weight_index];
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
  std::vector<double> v = best.sense_x;
  std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
  std::vector<double> v = best.sense_y;
  std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
