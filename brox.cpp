/**
  * Method as defined by Brox et al.

*/

#include "brox.hpp"
#define EPSILON 0.001


/**
  * Set up the parameters
*/
void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 100, 1000, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 15, 100, 10};
  parameter gamma = {"gamma", 500, 1000, 1000};
  parameter maxiter = {"maxiter", 200, 2000, 1};

  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
  parameters.insert(std::make_pair<std::string, parameter>(gamma.name, gamma));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
}


cv::Mat computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters){


}


double L1(double value){
  return std::sqrt(value + EPSILON);
}

double L1dot(double value){
  return 1.0/std::sqrt(value + EPSILON);
}
