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
  parameter maxlevel = {"maxlevel", 4, 10, 1};
  parameter wrapfactor = {"wrapfactor", 5, 10, 10};

  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
  parameters.insert(std::make_pair<std::string, parameter>(gamma.name, gamma));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
  parameters.insert(std::make_pair<std::string, parameter>(maxlevel.name, maxlevel));
  parameters.insert(std::make_pair<std::string, parameter>(wrapfactor.name, wrapfactor));
}


cv::Mat computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters){

  cv::Mat i1smoothed, i2smoothed, i1, i2, i2_wrap;
  int maxlevel = parameters.at("maxlevel").value;
  int maxiter = parameters.at("maxiter").value;
  double wrapfactor = (double)parameters.at("wrapfactor").value/parameters.at("wrapfactor").divfactor;
  double gamma = (double)parameters.at("gamma").value/parameters.at("gamma").divfactor;
  double sigma = (double)parameters.at("sigma").value/parameters.at("sigma").divfactor;
  double hx, hy;

  // make deepcopy, so images are untouched
  i1smoothed = image1.clone();
  i2smoothed = image2.clone();

  // convert to floating point images
  i1smoothed.convertTo(i1smoothed, CV_64F);
  i2smoothed.convertTo(i2smoothed, CV_64F);

  // blurring of the images (before resizing)
  cv::GaussianBlur(i1smoothed, i1smoothed, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2smoothed, i2smoothed, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // initialize parital and complete flowfield
  cv::Mat_<cv::Vec2d> partial(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield(i1smoothed.size());
  cv::Mat flowfield_wrap;

  // loop for over levels
  for (int k = maxlevel; k >= 0; k--){

    // set steps in x and y-direction with 1.0/wrapfactor^level
    hx = 1.0/std::pow(wrapfactor, k);
    hy = hx;

    // scale to level, using area resampling
    cv::resize(i1smoothed, i1, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);
    cv::resize(i2smoothed, i2, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);

    // resample flowfield to current level (for now using area resampling)
    cv::resize(flowfield, flowfield, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial, partial, i1.size(), 0, 0, cv::INTER_AREA);

    // wrap image 2 with current flowfield
    flowfield.convertTo(flowfield_wrap, CV_32FC2);
    cv::remap(i2, i2_wrap, flowfield_wrap, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT, cv::Scalar(0));
    i2 = i2_wrap;

    // compute tensors
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, hy, hx) + gamma * ComputeGradientTensor(i1, i2, hx, hy);

    // add 1px border to flowfield and partial
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial, partial, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    // set partial flowfield to zero
    partial = partial * 0;

    // main loop
    for (int i = 0; i < maxiter; i++){
      // call step function
      Brox_step(t, flowfield, partial, parameters, hx, hy);
    }

    // add partial flowfield to complete flowfield
    flowfield = flowfield + partial;
  }

  return flowfield(cv::Rect(1, 1, image1.cols, image1.rows));
}


/**
  * Inner loop of the Brox et al method
  * (for now only use spatial smoothness term)
*/
void Brox_step(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               std::unordered_map<std::string, parameter> &parameters,
               double hx,
               double hy){

  // get parameters
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;

  // helper variables
  double xm, xp, ym, yp;

  // update partial flow field
  for (int i = 1; i < p.rows - 1; i++){
    for (int j = 1; j < p.cols - 1; j++){

      // handle borders


    }
  }

}

double L1(double value){
  return (value < 0 ) ? 0 : std::sqrt(value + EPSILON);
}

double L1dot(double value){
  value = value < 0 ? 0 : value;
  return 1.0/std::sqrt(value + EPSILON);
}
