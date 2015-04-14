/**
  * Horn Schunck method with additional gradient constraint
*/

#include "hornschunck_separation.hpp"
const double DELTA=1.0;


void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 100, 1000, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 15, 100, 10};
  parameter gamma = {"gamma", 500, 1000, 1000};
  parameter maxiter = {"maxiter", 200, 400, 1};
  parameter wrapfactor = {"wrapfactor", 95, 100, 100};
  parameter nonlinear_step = {"nonlinear_step", 10, 150, 1};
  parameter kappa = {"kappa", 25, 100, 100};
  parameter beta = {"beta", 4, 1000, 100};
  parameter deltat = {"deltat", 25, 100, 100};
  parameter phi_iter = {"phi_iter", 10, 100, 1};
  parameter iter_flow_before_phi = {"iter_flow_before_phi", 10, 100, 1};


  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
  parameters.insert(std::make_pair<std::string, parameter>(gamma.name, gamma));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
  parameters.insert(std::make_pair<std::string, parameter>(kappa.name, kappa));
  parameters.insert(std::make_pair<std::string, parameter>(beta.name, beta));
  parameters.insert(std::make_pair<std::string, parameter>(deltat.name, deltat));
  parameters.insert(std::make_pair<std::string, parameter>(phi_iter.name, phi_iter));
  parameters.insert(std::make_pair<std::string, parameter>(iter_flow_before_phi.name, iter_flow_before_phi));
  parameters.insert(std::make_pair<std::string, parameter>(wrapfactor.name, wrapfactor));
  parameters.insert(std::make_pair<std::string, parameter>(nonlinear_step.name, nonlinear_step));

}


void computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters,
                         cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &phi){

  cv::Mat i1smoothed, i2smoothed, i1, i2;
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
  cv::Mat_<cv::Vec2d> partial_p(i1smoothed.size());
  cv::Mat_<cv::Vec2d> partial_m(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield_p(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield_m(i1smoothed.size());

  //flowfield.create(i1smoothed.size());
  cv::Mat flowfield_wrap;
  partial_p = cv::Vec2d(0,0);
  partial_m = cv::Vec2d(0,0);
  flowfield_p = cv::Vec2d(0,0);
  flowfield_m = cv::Vec2d(0,0);

  // make a 2-channel matrix with each pixel with its coordinates as value (serves as basis for flowfield remapping)
  cv::Mat remap_basis(image1.size(), CV_32FC2);
  for (int i = 0; i < image1.rows; i++){
   for (int j = 0; j < image1.cols; j++){
     remap_basis.at<cv::Vec2f>(i,j)[0] = (float)j;
     remap_basis.at<cv::Vec2f>(i,j)[1] = (float)i;
   }
  }

  // loop over levels
  for (int k = maxlevel; k >= 0; k--){
    std::cout << "Level: " << k << std::endl;

    // set steps in x and y-direction with 1.0/wrapfactor^level
    hx = 1.0/std::pow(wrapfactor, k);
    hy = hx;

    // scale to level, using area resampling
    cv::resize(i1smoothed, i1, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);
    cv::resize(i2smoothed, i2, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);

    // resample flowfield to current level (for now using area resampling)
    cv::resize(flowfield, flowfield, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(flowfield_p, flowfield_p, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(flowfield_m, flowfield_m, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial_p, partial_p, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial_m, partial_m, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(phi, phi, i1.size(), 0, 0, cv::INTER_AREA);

    flowfield = flowfield * wrapfactor;
    flowfield_p = flowfield_p * wrapfactor;
    flowfield_m = flowfield_m * wrapfactor;

    // set partial flowfield to zero
    partial_p = cv::Vec2d(0,0);
    partial_m = cv::Vec2d(0,0);

    // wrap image 2 with current flowfield
    flowfield.convertTo(flowfield_wrap, CV_32FC2);
    flowfield_wrap = flowfield_wrap + remap_basis(cv::Rect(0, 0, flowfield_wrap.cols, flowfield_wrap.rows));
    cv::remap(i2, i2, flowfield_wrap, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(0));


    // compute tensors
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, hy, hx) + gamma * ComputeGradientTensor(i1, i2, hx, hy);

    // add 1px border to flowfield, parital and tensor
    cv::copyMakeBorder(flowfield_p, flowfield_p, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(flowfield_m, flowfield_m, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial_p, partial_p, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial_m, partial_m, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(phi, phi, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);


    // main loop
    cv::Mat_<double> data_p(partial_p.size(), CV_64F);
    cv::Mat_<double> data_m(partial_p.size(), CV_64F);
    cv::Mat_<double> smooth_p(partial_p.size(), CV_64F);
    cv::Mat_<double> smooth_m(partial_p.size(), CV_64F);
    int nonlinear_step = parameters.at("nonlinear_step").value;
    for (int i = 0; i < maxiter; i++){
     if (i % nonlinear_step == 0 || i == 0){
       // computed terms dont have L1 norm yet
       computeDataTerm(partial_p, t, data_p);
       computeDataTerm(partial_m, t, data_m);
       computeSmoothnessTerm(flowfield_p, partial_p, smooth_p, hx, hy);
       computeSmoothnessTerm(flowfield_m, partial_m, smooth_m, hx, hy);
     }

     Brox_step_iso_smooth(t, flowfield_p, flowfield_m, partial_p, flowfield_m, data_p, data_m, smooth_p, smooth_m, phi, parameters, hx);
     updatePhi(data_p, data_m, smooth_p, smooth_m, phi, parameters, hx);
    }

    // add partial flowfield to complete flowfield
    flowfield_p = flowfield_p + partial_p;
    flowfield_m = flowfield_m + partial_m;
    for (int i = 0; i < flowfield_p.rows; i++){
      for (int j = 0; j < flowfield_p.cols; j++){
        flowfield(i,j) = (phi(i,j) > 0) ? flowfield_p(i,j) : flowfield_m(i,j);
      }
    }
  }

  cv::Mat f_p, f_m;
  computeColorFlowField(flowfield_p, f_p);
  computeColorFlowField(flowfield_m, f_m);
  cv::imshow("postive", f_p);
  cv::imshow("negative", f_m);
  //std::cout << phi << std::endl;
  flowfield = flowfield(cv::Rect(1,1,image1.cols, image1.rows));
  phi = phi(cv::Rect(1,1,image1.cols, image1.rows));
}


/**
  * this functions performs one iteration step in the hornschunck algorithm
  * @params &cv::Mat t Brightnesstensor for computation
  * @params &cv::Mat_<cv::Vec2d> flowfield The matrix for the flowfield which is computed
  * @params &std::unordered_map<std::string, parameter> parameters The parameter hash map for the algorithm
*/
void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &t,
                          const cv::Mat_<cv::Vec2d> &flowfield_p,
                          const cv::Mat_<cv::Vec2d> &flowfield_m,
                          cv::Mat_<cv::Vec2d> &partial_p,
                          cv::Mat_<cv::Vec2d> &partial_m,
                          const cv::Mat_<double> &data_p,
                          const cv::Mat_<double> &data_m,
                          const cv::Mat_<double> &smooth_p,
                          const cv::Mat_<double> &smooth_m,
                          const cv::Mat_<double> &phi,
                          const std::unordered_map<std::string, parameter> &parameters,
                          double h){


  updateU(flowfield_p, partial_p, phi, data_p, smooth_p, t, parameters, h, 1);
  updateU(flowfield_m, partial_m, phi, data_m, smooth_m, t, parameters, h, -1);

  updateV(flowfield_p, partial_p, phi, data_p, smooth_p, t, parameters, h, 1);
  updateV(flowfield_m, partial_m, phi, data_m, smooth_m, t, parameters, h, -1);

}


void updateU(cv::Mat_<cv::Vec2d> &flowfield,
             cv::Mat_<double> &phi,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             int sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;


  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++){

      if (phi(i,j)*sign > 0){
        // pixel is in the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h) * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0;
        xm =  (j > 1) * alpha/(h*h) * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0;
        yp =  (i < flowfield.rows-2) * alpha/(h*h) * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0;
        ym =  (i > 1) * alpha/(h*h) * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0;
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          - H(kappa * phi(i,j) *sign) * (t(i, j)[4] + t(i, j)[3] * flowfield(i,j)[1])
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(H(kappa * phi(i,j)*sign) * t(i, j)[0] + sum);


      } else {
        // for now use smoothess term here

        // test for borders
        xp =  (j < flowfield.cols-2) * 1.0/(h*h);
        xm =  (j > 1) * 1.0/(h*h);
        yp =  (i < flowfield.rows-2) * 1.0/(h*h);
        ym =  (i > 1) * 1.0/(h*h);
        sum = xp + xm + yp + ym;

        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(sum);

      }
    }
  }
}


void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h,
               int sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
   double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
   double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;


   for (int i = 1; i < flowfield.rows-1; i++){
     for (int j = 1; j < flowfield.cols-1; j++){

       if (phi(i,j)*sign > 0){
         // pixel is in the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h) * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0;
        xm =  (j > 1) * alpha/(h*h) * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0;
        yp =  (i < flowfield.rows-2) * alpha/(h*h) * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0;
        ym =  (i > 1) * alpha/(h*h) * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0;
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          - H(kappa * phi(i,j) *sign) * (t(i, j)[5] + t(i, j)[3] * flowfield(i,j)[0])
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(H(kappa * phi(i,j)*sign) * t(i, j)[1] + sum);

      } else {
        // pixel lies out of the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * 1.0/(h*h);
        xm =  (j > 1) * 1.0/(h*h);
        yp =  (i < flowfield.rows-2) * 1.0/(h*h);
        ym =  (i > 1) * 1.0/(h*h);
        sum = xp + xm + yp + ym;

        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(sum);

      }
    }
  }
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
      smooth(i,j) = tmp;
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
      data(i,j) = tmp;
    }
  }
}





/*

void updateU(cv::Mat_<cv::Vec2d> &flowfield,
             cv::Mat_<double> &phi,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             int sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;
  double H_s = 0;

  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++){

      //if (phi(i,j)*sign > 0){
        // pixel is in the segment
        H_s = (phi(i,j) * sign > 0) ? 1 : 0;

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h);
        xm =  (j > 1) * alpha/(h*h);
        yp =  (i < flowfield.rows-2) * alpha/(h*h) ;
        ym =  (i > 1) * alpha/(h*h);
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          - H_s * (t(i, j)[4] + t(i, j)[3] * flowfield(i,j)[1])
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(H_s * t(i, j)[0] + sum);
    }
  }
}


void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h,
               int sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
   double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
   double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;
   double H_s = 0;
   for (int i = 1; i < flowfield.rows-1; i++){
     for (int j = 1; j < flowfield.cols-1; j++){

       H_s = (phi(i,j) * sign > 0) ? 1 : 0;
         // pixel is in the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h);
        xm =  (j > 1) * alpha/(h*h);
        yp =  (i < flowfield.rows-2) * alpha/(h*h);
        ym =  (i > 1) * alpha/(h*h);
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          - H_s * (t(i, j)[5] + t(i, j)[3] * flowfield(i,j)[0])
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(H_s * t(i, j)[1] + sum);


    }
  }
}

*/

void updatePhi(cv::Mat_<cv::Vec2d> &flowfield_p,
               cv::Mat_<cv::Vec2d> &flowfield_m,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h){

  // update the segment indicator function using implicit scheme

  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double beta = (double)parameters.at("beta").value/parameters.at("beta").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double deltat = (double)parameters.at("deltat").value/parameters.at("deltat").divfactor;

  double data, smooth, phi_update, c1, c2, c3, c4, m, c, tmp;


  // compute derivatives (not very efficient, we could use one mat for each derivatives)
  cv::Mat_<double> f_p[2], f_m[2];
  cv::Mat_<double> ux_p, ux_m, uy_p, uy_m, vx_p, vx_m, vy_p, vy_m, phix, phiy, kernel;

  // split flowfields
  split(flowfield_p, f_p);
  split(flowfield_m, f_m);


  kernel = (cv::Mat_<double>(1,3) << -1, 0, 1);
  cv::filter2D(f_p[0], ux_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[0], ux_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_p[1], vx_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[1], vx_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(phi, phix, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);


  kernel = (cv::Mat_<double>(3,1) << -1, 0, 1);
  cv::filter2D(f_p[0], uy_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[0], uy_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_p[1], vy_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[1], vy_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(phi, phiy, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  for (int i = 1; i < phi.rows-1; i++){
    for (int j = 1; j < phi.cols-1; j++){

      // compute the data term parts
      smooth = ux_p(i,j) * ux_p(i,j) + uy_p(i,j) * uy_p(i,j) + vx_p(i,j) * vx_p(i,j) + vy_p(i,j) * vy_p(i,j);
      smooth = smooth - ux_m(i,j) * ux_m(i,j) - uy_m(i,j) * uy_m(i,j) - vx_m(i,j) * vx_m(i,j) - vy_m(i,j) * vy_m(i,j);


      // compute smoothness term parts
      data =   t(i,j)[0] * f_p[0](i,j) * f_p[0](i,j)        // J11*du^2
             + t(i,j)[1] * f_p[1](i,j) * f_p[1](i,j)        // J22*dv^2
             + t(i,j)[2]                                    // J33
             + t(i,j)[3] * f_p[0](i,j) * f_p[1](i,j) * 2    // J21*du*dv
             + t(i,j)[4] * f_p[0](i,j) * 2                  // J13*du
             + t(i,j)[5] * f_p[1](i,j) * 2;                 // J23*dv

      data = data - t(i,j)[0] * f_m[0](i,j) * f_m[0](i,j)        // J11*du^2
                  - t(i,j)[1] * f_m[1](i,j) * f_m[1](i,j)        // J22*dv^2
                  - t(i,j)[2]                                    // J33
                  - t(i,j)[3] * f_m[0](i,j) * f_m[1](i,j) * 2    // J21*du*dv
                  - t(i,j)[4] * f_m[0](i,j) * 2                  // J13*du
                  - t(i,j)[5] * f_m[1](i,j) * 2;                 // J23*dv


      // terms with phi using semi-implicit scheme
      // check on bondaries
      /*
      xm = (j > 1) * 0.5/(h*h) * phi_norm(phix,phiy,i,j,0,-1);
      xp = (j < phi.cols-2) * 0.5/(h*h) * phi_norm(phix,phiy,i,j,0,1);
      ym = (i > 1) * 0.5/(h*h) * phi_norm(phix,phiy,i,j,-1,0);
      yp = (i < phi.rows-2) * 0.5/(h*h) * phi_norm(phix,phiy,i,j,1,0);
      sum = xm + xp + ym + yp;

      phi_update =  xm * (phi(i,j-1))
                  + xp * (phi(i,j+1))
                  + ym * (phi(i-1,j))
                  + yp * (phi(i+1,j));


      // update phi
      phi(i,j) = phi(i,j) + deltat * (
                  beta * Hdot(phi(i,j)) * phi_update
                - kappa * Hdot(kappa * phi(i,j)) * data
                - alpha * Hdot(phi(i,j)) * smooth);
      phi(i,j) = phi(i,j)/(1.0+beta*Hdot(phi(i,j))*sum*deltat);*/

      // using the vese chan discretization
      tmp = (j< phi.cols-2) * std::pow((phi(i,j+1) - phi(i,j))/h, 2) + (i>1)*(i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i-1,j))/(2*h),2);
      c1 = (tmp == 0) ? 0 : std::sqrt(1.0/tmp);

      tmp = (j>1)*std::pow((phi(i,j) - phi(i,j-1))/h, 2) + (i<phi.rows-2)*(i>1)*(j>1)*std::pow((phi(i+1,j-1) - phi(i-1,j-1))/(2*h),2);
      c2 = (tmp == 0) ? 0 : std::sqrt(1.0/tmp);

      tmp = (j>1)*(j<phi.cols-2)*std::pow((phi(i,j+1) - phi(i,j-1))/(2*h), 2) + (i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i,j))/(h),2);
      c3 = (tmp == 0) ? 0 : std::sqrt(1.0/tmp);

      tmp = (i>1)*(j>1)*(j<phi.cols-2)*std::pow((phi(i-1,j+1) - phi(i-1,j-1))/(2*h), 2) + (i>1)*std::pow((phi(i,j) - phi(i-1,j))/(h),2);
      c4 = (tmp == 0) ? 0 : std::sqrt(1.0/tmp);

      m = (deltat*Hdot(phi(i,j))*beta)/(h*h);
      c = 1+m*(c1+c2+c3+c4);
      phi(i,j) = (1.0/c)*(phi(i,j) + m*(c1*phi(i,j+1)+c2*phi(i,j-1)+c3*phi(i+1,j)+c4*phi(i-1,j))
                          -deltat*kappa*Hdot(kappa*phi(i,j))*data
                          -deltat*alpha*Hdot(phi(i,j))*smooth);
    }
  }

}

double phi_norm(cv::Mat_<double> &phix, cv::Mat_<double> &phiy, int i, int j, int offset_i, int offset_j){
  double tmp = phix(i,j)*phix(i,j) + phiy(i,j)*phiy(i,j);
  double tmp2 = phix(i+offset_i,j+offset_j) * phix(i+offset_i,j+offset_j) + phiy(i+offset_i,j+offset_j) * phiy(i+offset_i,j+offset_j);
  if (tmp <= 0.0 || tmp2 <= 0.0){ return 0; }
  return std::sqrt(1.0/tmp)+std::sqrt(1.0/tmp2);
}

double H(double x){
  return 0.5 * (1 + (2.0/M_PI)*std::atan(x/DELTA));
}

double Hdot(double x){
  return (1.0/M_PI) * (DELTA/(DELTA*DELTA + x*x));
}
