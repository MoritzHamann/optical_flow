#include "misc.hpp"

/**
  * computes a color representation of the current flow
  * @param cv::Mat f Current Flowfield
  * @return cv::Mat A RGB image of the flowfield, visualized with hsv conversion
*/
void computeColorFlowFieldError(const cv::Mat_<cv::Vec2d> &f, cv::Mat &img){
  // helper Mats
  cv::Mat flowfieldcomponents[2];
  cv::Mat magnitude, angle, hsv, _hsv[3];
  _hsv[2] = cv::Mat::ones(f.size(), CV_64F) * 255;

  // compute color image from flowfield using hsv
  split(f, flowfieldcomponents);
  cv::cartToPolar(flowfieldcomponents[0], flowfieldcomponents[1], magnitude, angle, true);
  double max = 1;
  cv::minMaxIdx(magnitude, NULL, &max);
  if (max > 0){
    magnitude = magnitude * 1.0/max;
  }
  _hsv[0] = angle;
  _hsv[1] = magnitude;
  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv, CV_32FC3);     // cannot convert directly to CV_8UC3, because max(angle) could be 360 degrees
  cvtColor(hsv, hsv, CV_HSV2BGR);
  hsv.convertTo(img, CV_8UC3);
}

void computeColorFlowField(const cv::Mat_<cv::Vec2d> &f, cv::Mat &img){
  // helper Mats
  cv::Mat flowfieldcomponents[2];
  cv::Mat magnitude, angle, hsv, _hsv[3];
  _hsv[1] = cv::Mat::ones(f.size(), CV_64F);

  // compute color image from flowfield using hsv
  split(f, flowfieldcomponents);
  cv::cartToPolar(flowfieldcomponents[0], flowfieldcomponents[1], magnitude, angle, true);
  double max = 1;
  cv::minMaxIdx(magnitude, NULL, &max);
  if (max > 0){
    magnitude = magnitude * 255.0/max;
  }
  _hsv[0] = angle;
  _hsv[2] = magnitude;
  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv, CV_32FC3);     // cannot convert directly to CV_8UC3, because max(angle) could be 360 degrees
  cvtColor(hsv, hsv, CV_HSV2BGR);
  hsv.convertTo(img, CV_8UC3);
}


/**
  * reads the groundtruth - stored in filename - into a vectorfield
  * @param std::string filename The filename of the groundtruth file
  * @return cv::Mat The groundtruh as a vector field
*/
void loadBarronFile(std::string filename, cv::Mat_<cv::Vec2d> &truth){

  FILE *file = fopen(filename.c_str(), "r");
  if (file == NULL){
    std::cout << "Error opening Barron File: " << filename << "! Exit" << std::endl;
    std::exit(1);
  }

  // read the header information
  float help;
  std::fread (&help, sizeof(float), 1, file);
  int nx_and_offsetx  = (int) help;
  std::fread (&help, sizeof(float), 1, file);
  int ny_and_offsety  = (int) help;
  std::fread (&help, sizeof(float), 1, file);
  int nx  = (int) help;
  std::fread (&help, sizeof(float), 1, file);
  int ny  = (int) help;
  std::fread (&help, sizeof(float), 1, file);
  int offsetx = (int) help;
  std::fread (&help, sizeof(float), 1, file);
  int offsety = (int) help;

  // initialize truth vector field
  truth.create(ny, nx);

  // make tmp array
  std::vector< std::vector<double> > tmpu(nx_and_offsetx, std::vector<double>(ny_and_offsety));
  std::vector< std::vector<double> > tmpv(nx_and_offsetx, std::vector<double>(ny_and_offsety));

  // read complete data
  for (int j = 0; j < ny_and_offsety; j++){
    for (int i = 0; i < nx_and_offsetx; i++){
      fread(&help, sizeof(float), 1, file);
      tmpu[i][j] = (double)help;

      fread(&help, sizeof(float), 1, file);
      tmpv[i][j] = (double)help;
    }
  }
  fclose(file);

  // set data without crop
  for (int j = 0; j < ny; j++){
    for (int i = 0; i < nx; i++){
      truth(j,i)[0] = tmpu[i + offsetx][j + offsety];
      truth(j,i)[1] = tmpv[i + offsetx][j + offsety];
    }
  }

  //return truth;
}


void TrackbarCallback(int value, void *userdata){
  parameter *p = static_cast<parameter*>(userdata);
  std::cout << p->name << ": " << std::floor((double)p->value*100/p->divfactor)/100 << std::endl;
}






void computeColorFlowField2(const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat &img){

  //cv::Mat_<cv::Vec2d> flowfield = f;

  // make temporary array
  //cv::Mat img(flowfield.size(), CV_8UC3);

  double Pi = 3.141592653589793238463;;
  double amp;
  double phi;
  double alpha, beta;
  double x, y;

  for (int i = 0; i < flowfield.rows; i++){
    for (int j = 0; j < flowfield.cols; j++){
      x = flowfield(i, j)[0];
      y = flowfield(i, j)[1];

      /* determine amplitude and phase (cut amp at 1) */
      amp = sqrt (x * x + y * y);
      if (amp > 1) amp = 1;
      if (x == 0.0)
        if (y >= 0.0) phi = 0.5 * Pi;
        else phi = 1.5 * Pi;
      else if (x > 0.0)
        if (y >= 0.0) phi = atan (y/x);
        else phi = 2.0 * Pi + atan (y/x);
      else phi = Pi + atan (y/x);
      phi = phi / 2.0;

      // interpolation between red (0) and blue (0.25 * Pi)
      if ((phi >= 0.0) && (phi < 0.125 * Pi)) {
        beta  = phi / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha *   0.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha *   0.0 + beta * 255.0));
      }
      if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
        beta  = (phi-0.125 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 255.0 + beta *  64.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha *   0.0 + beta *  64.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
      }
      // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
      if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
        beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha *  64.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha *  64.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
      }
      if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
        beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha *   0.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 255.0 + beta *   0.0));
      }
      // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
      if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
        beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 0.0   + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 0.0   + beta * 0.0));
      }
      // interpolation between yellow (0.75 * Pi) and red (Pi)
      if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
        beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha * 255.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 0.0   + beta *   0.0));
      }

      /* check RGB range */
      /*img.at<cv::Vec3b>(i,j)[0] = byte_range((int)img[i][j].r);
      img.at<cv::Vec3b>(i,j)[1] = byte_range((int)img[i][j].g);
      img.at<cv::Vec3b>(i,j)[2] = byte_range((int)img[i][j].b);*/
    }
  }

  cvtColor(img, img, CV_RGB2BGR);
  //return img;
}






double CalcAngularError(const cv::Mat_<cv::Vec2d> &flowfield, const cv::Mat_<cv::Vec2d> truth){
  double amount = 0;
  double tmp1 = 0;
  double tmp2 = 0;
  double sum_l2 = 0;
  double sum_ang = 0;

  for (int i = 0; i < flowfield.rows; i++) {
    for (int j = 0; j < flowfield.cols; j++){
      // test if reference flow vector exists
      if (truth(i,j)[0] != 100.0 || truth(i,j)[1] != 100.0){
        amount++;
        tmp1 = truth(i,j)[0] - flowfield(i,j)[0];
        tmp2 = truth(i,j)[1] - flowfield(i,j)[1];

        sum_l2 += sqrt(tmp1*tmp1 + tmp2*tmp2);
        tmp1 = (truth(i,j)[0] * flowfield(i,j)[0] + truth(i,j)[1] * flowfield(i,j)[1] + 1.0)
   		     / sqrt( (truth(i,j)[0] * truth(i,j)[0] + truth(i,j)[1] * truth(i,j)[1] + 1.0)
   			     * (flowfield(i,j)[0] * flowfield(i,j)[0] + flowfield(i,j)[1] * flowfield(i,j)[1] + 1.0) );

        if (tmp1 > 1.0) tmp1 = 1.0;
        if (tmp1 < - 1.0) tmp1 = - 1.0;
        sum_ang += acos(tmp1) * 180.0/M_PI;
      }
    }
  }

  return sum_ang / amount;
}


void computeSegmentationImage(const cv::Mat_<double> &phi, const cv::Mat_<uchar> &image1, cv::Mat &segmentation){
  segmentation.create(phi.size(), CV_8U);
  for (int i = 0; i < phi.rows; i++){
    for (int j = 0; j < phi.cols; j++){
      segmentation.at<uchar>(i,j) = (phi(i,j) > 0) ? 254 : 0;
      if (std::isnan(segmentation.at<uchar>(i,j)) || !std::isfinite(segmentation.at<uchar>(i,j))){
        std::cout << "error segementation" << std::endl;
      }
    }
  }
}

void remap_border(cv::Mat &image, const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &mask){
  // wrap the image with the information in flowfield. if information is out of boundaries, set mask to 0 (then the data term will not be taken into account)

  double x_float, y_float, x_int, y_int, x_delta, y_delta, value;
  
  // copy input image to temporary image
  cv::Mat image2 = image.clone();

  // add border to image2 for simplicity
  cv::copyMakeBorder(image2, image2, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);

  for (int i = 1; i < flowfield.rows - 1; i++){
    for (int j = 1; j < flowfield.cols - 1; j++){
      
      // compute new floating point location of image (we don't need to rescale with h, because flowfield is already rescaled before)
      x_float = j + flowfield(i,j)[0];
      y_float = i + flowfield(i,j)[1];
      
      if ( (x_float < 1) || (x_float > flowfield.cols-2) || (y_float < 1) || (y_float > flowfield.rows-2) ){
        // flow is outsite of boundaries

        mask(i,j) = 0;
        value = image2.at<double>(i,j);

      } else {
        // flow is inside boundaries => wraping with bilinear interpolation
        mask(i,j) = 1.0;
        
        // compute integral and non-integral part of flow
        x_int = std::floor(x_float);
        y_int = std::floor(y_float);
        x_delta = x_float - x_int;
        y_delta = y_float - y_int;


        // perform bilinear interpolation
        value = (1.0-x_delta) * (1.0-y_delta) * image2.at<double>(y_int  , x_int  ) +
                (1.0-x_delta) *      y_delta  * image2.at<double>(y_int+1, x_int  ) +
                     x_delta  * (1.0-y_delta) * image2.at<double>(y_int  , x_int+1) +
                     x_delta  *      y_delta  * image2.at<double>(y_int+1, x_int+1);
      }

      image.at<double>(i-1, j-1) = value;
    }
  }
}
