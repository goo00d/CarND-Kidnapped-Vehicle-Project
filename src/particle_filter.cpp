/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if(is_initialized) return;
	is_initialized = true;
	num_particles = 100;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i < num_particles; ++i) {
	double sample_x, sample_y, sample_theta;
	sample_x = dist_x(gen);
	sample_y = dist_y(gen);
	sample_theta = dist_theta(gen);
	Particle particle;
	particle.id = i;
	particle.x = sample_x;
	particle.y = sample_y;
	particle.theta = sample_theta;
	particle.weight = 1.0;
	particles.push_back(particle);
	
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	if(fabs(yaw_rate)<10E-8){
		normal_distribution<double> dist_x(0, std_pos[0]);
		normal_distribution<double> dist_y(0, std_pos[1]);
		default_random_engine gen;
		for(int i = 0;i<num_particles;i++)
		{
			double x = particles[i].x;
			double y = particles[i].y;
			double theta = particles[i].theta;
			double xf = x + velocity*delta_t*cos(theta);
			double yf = y + velocity*delta_t*sin(theta);
			particles[i].x = xf+dist_x(gen);
			particles[i].y = yf+dist_y(gen);
		}
	}
	else{
		double vpt = velocity/yaw_rate;
		normal_distribution<double> dist_x(0, std_pos[0]);
		normal_distribution<double> dist_y(0, std_pos[1]);
		normal_distribution<double> dist_theta(0, std_pos[2]);
		default_random_engine gen;
		for(int i = 0;i<num_particles;i++)
		{
			double x = particles[i].x;
			double y = particles[i].y;
			double theta = particles[i].theta;
			
			double xf = x + vpt*(sin(theta+yaw_rate*delta_t)-sin(theta));
			double yf = y + vpt*(-cos(theta+yaw_rate*delta_t)+cos(theta));
			double thetaf = theta+ yaw_rate*delta_t;
			particles[i].x = xf+dist_x(gen);
			particles[i].y = yf+dist_y(gen);
			particles[i].theta = thetaf+dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0;i<observations.size();i++)
	{
		double mindist = 10.0e8;
		double minIndex = 0;
		double x = observations[i].x;
		double y = observations[i].y;
		for(int j = 0;j<predicted.size();j++)
		{
			double dist2pt = dist(x,y,predicted[j].x,predicted[j].y);
			if(dist2pt<mindist)
			{
				mindist = dist2pt;
				minIndex = j;
			}
		}
		observations[i].x = predicted[minIndex].x;
		observations[i].y = predicted[minIndex].y;
	}
}
void translocal2world(double& worldx,double &worldy,double localx,double localy,double transx,double transy,double transang)
{
	worldx = localx*cos(transang)-localy*sin(transang)+transx;
	worldy = localx*sin(transang)+localy*cos(transang)+transy;
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	std::vector<LandmarkObs> landmap;
	for(int i = 0;i<map_landmarks.landmark_list.size();i++)
	{
		LandmarkObs tmp;
		tmp.x = map_landmarks.landmark_list[i].x_f;
		tmp.y = map_landmarks.landmark_list[i].y_f;
		tmp.id = map_landmarks.landmark_list[i].id_i;
		landmap.push_back(tmp);
	}
	double pi = 2*asin(1.0);
	double sigmax = std_landmark[0];
	double sigmay = std_landmark[1];
	for(int i = 0;i<particles.size();i++)
	{
		double transx = particles[i].x;
		double transy = particles[i].y;
		double transang = particles[i].theta;
		std::vector<LandmarkObs> observationworld;
		for(int j = 0;j<observations.size();j++)
		{
			LandmarkObs tmp;
			translocal2world(tmp.x,tmp.y,observations[j].x,observations[j].y,transx,transy,transang);
			if(dist(transx,transy,tmp.x,tmp.y)>sensor_range) continue;
			observationworld.push_back(tmp);
		}
		std::vector<LandmarkObs> observationworldpredicted = observationworld;
		dataAssociation(landmap,observationworldpredicted);
		double weight = 1.0;
		for(int k = 0;k<observationworld.size();k++)
		{
			double &xobs = observationworld[k].x;
			double &xpred = observationworldpredicted[k].x;
			double &yobs = observationworld[k].y;
			double &ypred = observationworldpredicted[k].y;
			weight *= 1/2.0/pi/sigmax/sigmay*exp(-(xobs-xpred)*(xobs-xpred)/2/sigmax/sigmax-(yobs-ypred)*(yobs-ypred)/2/sigmay/sigmay);
		}
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	weights.clear();
	for(int i = 0;i<num_particles;i++)
	{
	 	weights.push_back(particles[i].weight);
	}
	default_random_engine gen;
	std::discrete_distribution<> d(weights.begin(),weights.end());
	
	std::vector<Particle> newparticles;
	for(int i = 0;i<num_particles;i++)
	{
		int index = d(gen);
		auto tmp = particles[index];
		newparticles.push_back(tmp);
	}
	particles = newparticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
