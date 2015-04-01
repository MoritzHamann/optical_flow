#ifndef BROX_HPP
#define BROX_HPP

#include "types.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
#include "tensor_computation.hpp"

void setupParameters(std::unordered_map<std::string, parameter> &parameters);
cv::Mat computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters);
void Brox_step(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               std::unordered_map<std::string, parameter> &parameters,
               double hx,
               double hy);

void Brox_step_wo(const cv::Mat_<cv::Vec6d> &t,
              const cv::Mat_<cv::Vec2d> &f,
              cv::Mat_<cv::Vec2d> &p,
              std::unordered_map<std::string, parameter> &parameters,
              double hx,
              double hy);

cv::Mat computeSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, cv::Mat_<cv::Vec2d> &p, double hx, double hy);
cv::Mat computeDataTerm(cv::Mat_<cv::Vec2d> &p, const cv::Mat_<cv::Vec6d> &t);
double L1(double value);
double L1dot(double value);

#endif
