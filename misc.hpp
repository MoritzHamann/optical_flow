#ifndef MISC_HPP
#define MISC_HPP

#include "types.hpp"

void computeColorFlowField(const cv::Mat_<cv::Vec2d> &f, cv::Mat &img);
void loadBarronFile(std::string filename, cv::Mat_<cv::Vec2d> &truth);
void TrackbarCallback(int value, void *userdata);
void computeColorFlowField2(const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat &img);
double CalcAngularError(const cv::Mat_<cv::Vec2d> &flowfield, const cv::Mat_<cv::Vec2d> truth);

void setupParameters(std::unordered_map<std::string, parameter> &parameters);
void computeFlowField(const cv::Mat &image1,
                 const cv::Mat &image2,
                 std::unordered_map<std::string, parameter> &parameters,
                 cv::Mat_<cv::Vec2d> &flowfield);
void computeFlowField(const cv::Mat &image1,
                 const cv::Mat &image2,
                 std::unordered_map<std::string, parameter> &parameters,
                 cv::Mat_<cv::Vec2d> &flowfield,
                 cv::Mat_<double> &phi);

#endif
