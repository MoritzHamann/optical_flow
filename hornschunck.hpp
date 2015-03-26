#ifndef HORNSCHUNCK_HPP
#define HORNSCHUNCK_HPP

#include "types.hpp"
#include "tensor_computation.hpp"
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <iostream>


void setupParameters(std::unordered_map<std::string, parameter> &parameters);
cv::Mat computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters);
void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t, cv::Mat_<cv::Vec2d> &flowfield, const std::unordered_map<std::string, parameter> &parameters);


#endif
