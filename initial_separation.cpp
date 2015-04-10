#include "initial_separation.hpp"


void initial_separation(const cv::Mat &image1,
                        const cv::Mat &image2,
                        cv::Mat_<double> &phi,
                        const std::unordered_map<std::string, parameter> &parameters,
                      ){

  // compute the initial separation with isotropic brox et al
  cv::Mat_<cv::Vec2d> flowfield(image1.size());
  computeInitialFlowfield(image1, image2, flowfield, parameters);
  segmentFlowfield(flowfield, phi, parameters);
}




void computeInitialFlowfield(const cv::Mat &image1,
                             const cv::Mat &image2,
                             cv::Mat_<cv::Vec2d> &flowfield,
                             const std::unordered_map<std::string, parameter> &parameters){

  cv::Mat i1smoothed, i2smoothed, i1, i2;
  int maxlevel = parameters.at("maxlevel_sep").value;
  int maxiter = parameters.at("maxiter_sep").value;
  double wrapfactor = (double)parameters.at("wrapfactor_sep").value/parameters.at("wrapfactor").divfactor;
  double gamma = (double)parameters.at("gamma_sep").value/parameters.at("gamma").divfactor;
  double sigma = (double)parameters.at("sigma_sep").value/parameters.at("sigma").divfactor;
  double hx, hy;

  // make deepcopy, so images are untouched
  i1smoothed = image1.clone();
  i2smoothed = image2.clone();

  // convert to floating point images
  i1smoothed.convertTo(i1smoothed, CV_64F);
  i2smoothed.convertTo(i2smoothed, CV_64F);

  // initialize parital and complete flowfield
  cv::Mat_<cv::Vec2d> partial(i1smoothed.size());
  flowfield.create(i1smoothed.size());
  cv::Mat flowfield_wrap;
  partial = cv::Vec2d(0,0);
  flowfield = cv::Vec2d(0,0);

  // make a 2-channel matrix with each pixel with its coordinates as value (serves as basis for flowfield remapping)
  cv::Mat remap_basis(image1.size(), CV_32FC2);
  for (int i = 0; i < image1.rows; i++){
    for (int j = 0; j < image1.cols; j++){
      remap_basis.at<cv::Vec2f>(i,j)[0] = (float)j;
      remap_basis.at<cv::Vec2f>(i,j)[1] = (float)i;
    }
  }

  // loop for over levels
  for (int k = maxlevel; k >= 0; k--){
    std::cout << "Segmentation Level: " << k << std::endl;

    // set steps in x and y-direction with 1.0/wrapfactor^level
    hx = 1.0/std::pow(wrapfactor, k);
    hy = hx;

    // scale to level, using area resampling
    cv::resize(i1smoothed, i1, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);
    cv::resize(i2smoothed, i2, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);

    // resample flowfield to current level (for now using area resampling)
    cv::resize(flowfield, flowfield, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial, partial, i1.size(), 0, 0, cv::INTER_AREA);

    flowfield = flowfield * wrapfactor;

    // wrap image 2 with current flowfield
    flowfield.convertTo(flowfield_wrap, CV_32FC2);
    flowfield_wrap = flowfield_wrap + remap_basis(cv::Rect(0, 0, flowfield_wrap.cols, flowfield_wrap.rows));
    cv::remap(i2, i2, flowfield_wrap, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(0));


    // compute tensors
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, hy, hx) + gamma * ComputeGradientTensor(i1, i2, hx, hy);

    // add 1px border to flowfield, parital and tensor
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial, partial, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    // set partial flowfield to zero
    partial = cv::Vec2d(0,0);

    // main loop
    cv::Mat_<double> data(partial.size(), CV_64F);
    cv::Mat_<double> smooth(partial.size(), CV_64F);
    int nonlinear_step = parameters.at("nonlinear_step").value;
    for (int i = 0; i < maxiter; i++){
      if (i % nonlinear_step == 0 || i == 0){
        computeDataTerm(partial, t, data);
        computeSmoothnessTerm(flowfield, partial, smooth, hx, hy);
      }
      Separation_step(t, flowfield, partial, data, smooth, parameters, hx, hy);
    }

    // add partial flowfield to complete flowfield
    flowfield = flowfield + partial;
  }

  flowfield = flowfield(cv::Rect(1, 1, image1.cols, image1.rows));
}



void segementFlowfield(cv::Mat_<cv::Vec2d> &f, cv::Mat_<double> &phi, const std::unordered_map<std::string, parameter> &parameters){

  // helper variables
  cv::Mat_<cv::Vec6d> affine;
  int blocksize = parameters.at("blocksize_sep").value;


  // estimated affine parameters for each block

}


void computeSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<double> &smooth, double hx, double hy){
  cv::Mat fc[2], pc[2];
  cv::Mat flow_u, flow_v, ux, uy, vx, vy, kernel;
  double tmp=0;

  // split flowfield in components
  cv::split(f, fc);
  flow_u = fc[0];
  flow_v = fc[1];

  // split partial flowfield in components
  cv::split(p, pc);
  flow_u = flow_u + pc[0];
  flow_v = flow_v + pc[1];

  //std::cout << flow_u.at<cv::Vec2d>(10,10) << ":" << flow_v.at<cv::Vec2d>(10,10) << std::endl;

  // derivates in y-direction
  kernel = (cv::Mat_<double>(3,1) << -1, 0, 1);
  cv::filter2D(flow_u, uy, CV_64F, kernel * 1.0/(2*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(flow_v, vy, CV_64F, kernel * 1.0/(2*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  // derivates in x-dirction
  kernel = (cv::Mat_<double>(1,3) << -1, 0, 1);
  cv::filter2D(flow_u, ux, CV_64F, kernel * 1.0/(2*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(flow_v, vx, CV_64F, kernel * 1.0/(2*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  for (int i = 0; i < p.rows; i++){
    for (int j = 0; j < p.cols; j++){
      tmp = ux.at<double>(i,j) * ux.at<double>(i,j) + uy.at<double>(i,j) * uy.at<double>(i,j);
      tmp = tmp + vx.at<double>(i,j) * vx.at<double>(i,j) + vy.at<double>(i,j) * vy.at<double>(i,j);
      smooth(i,j) = L1dot(tmp);
    }
  }
}




void computeDataTerm(const cv::Mat_<cv::Vec2d> &p, const cv::Mat_<cv::Vec6d> &t, cv::Mat_<double> &data){
  double tmp;

  for (int i= 0; i < p.rows; i++){
    for (int j = 0; j < p.cols; j++){
      tmp =   t(i,j)[0] * p(i,j)[0] * p(i,j)[0]         // J11*du^2
            + t(i,j)[1] * p(i,j)[1] * p(i,j)[1]         // J22*dv^2
            + t(i,j)[2]                                 // J33
            + t(i,j)[3] * p(i,j)[0] * p(i,j)[1] * 2     // J21*du*dv
            + t(i,j)[4] * p(i,j)[0] * 2                 // J13*du
            + t(i,j)[5] * p(i,j)[1] * 2;                // J23*dv
      data(i,j) = L1dot(tmp);
    }
  }
}


double L1(double value){
  return (value < 0 ) ? 0 : std::sqrt(value + EPSILON);
}


double L1dot(double value){
  value = value < 0 ? 0 : value;
  return 1.0/(2.0 * std::sqrt(value + EPSILON));
}
