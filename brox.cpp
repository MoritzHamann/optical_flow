/**
  * Method of Brox et all
*/

void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 100, 1000, 1};
  parameter omega = {"omega", 195, 2000, 100};
  parameter sigma = {"sigma", 15, 100, 10};
  parameter gamma = {"gamma", 100, 1000, 1};
  parameter maxiter = {"maxiter", 200, 1000, 1};

  parameters.insert(std::make_pair<std::string, paramter>(alpha.name, alpha.value));
  parameters.insert(std::make_pair<std::string, paramter>(omega.name, omega.value));
  parameters.insert(std::make_pair<std::string, paramter>(sigma.name, sigma.value));
  parameters.insert(std::make_pair<std::string, paramter>(gamma.name, gamma.value));
  parameters.insert(std::make_pair<std::string, paramter>(maxiter.name, maxiter.value));
}


cv::Mat computeFlowField(cv::Mat &image1, cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters){

  // convert images into 64 bit floating point images
  cv::Mat i1, i2;
  i1 = image1.clone();
  i1.convertTo(i2, CV_64F);
  i2 = image1.clone();
  i2.convertTo(i2, CV_64F);

  // Blurring
  double sigma = (double)parameters.at("sigma").value/parameters.at("sigma").divfactor;
  cv::GaussianBlur(i1, i1, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2, i2, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // compute Brightness and Gradient Tensors
  cv::Mat_<cv::Vec6d> tb = ComputeBrightnessTensor(i1, i2, 1, 1);
  cv::Mat_<cv::Vec6d> tg = ComputeGradientTensor(i1, i2, 1, 1);



  cv::Mat_<cv::Vec2d> flowfield()
}
