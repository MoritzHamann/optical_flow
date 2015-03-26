#ifndef BROX_HPP
#define BROX_HPP


void setupParameters(std::unordered_map<std::string, parameter> &parameters);
cv::Mat computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters);
double L1(double value);
double L1dot(double value);

#endif
