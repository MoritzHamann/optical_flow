#ifndef INITIAL_SEPARATION_HPP
#define INITIAL_SEPARATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "types.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
#include "tensor_computation.hpp"
#include "misc.hpp"


void initial_segmentation(const cv::Mat_<cv::Vec2d> &flowfield,
                        cv::Mat_<double> &phi,
                        const std::unordered_map<std::string, parameter> &parameters
                      );
void segementFlowfield(const cv::Mat_<cv::Vec2d> &f, cv::Mat_<double> &phi, const std::unordered_map<std::string, parameter> &parameters);

#endif
