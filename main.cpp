#include <iostream>
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "hornschunck.hpp"

#define WINDOW_NAME "optical flow"



/**
  * computes a color representation of the current flow
  * @param cv::Mat f Current Flowfield
  * @return cv::Mat A RGB image of the flowfield, visualized with hsv conversion
*/
cv::Mat computeColorFlowField(cv::Mat f){
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
  cvtColor(hsv, hsv, CV_HSV2RGB);
  hsv.convertTo(hsv, CV_8UC3);

  return hsv;
}


/**
  * reads the groundtruth - stored in filename - into a vectorfield
  * @param std::string filename The filename of the groundtruth file
  * @return cv::Mat The groundtruh as a vector field
*/
cv::Mat loadBarronFile(std::string filename){

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
  cv::Mat_<cv::Vec2d> truth(ny, nx);

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

  return truth;
}


void TrackbarCallback(int value, void *userdata){
  parameter *p = static_cast<parameter*>(userdata);
  std::cout << p->name << ": " << std::floor((double)p->value*100/p->divfactor)/100 << std::endl;
}






cv::Mat computeColorFlowField2(cv::Mat_<cv::Vec2d> &flowfield){

  // make temporary array
  cv::Mat img(flowfield.size(), CV_8UC3);

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
  return img;
}






double CalcAngularError(cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<cv::Vec2d> truth){
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









int main(int argc, char *argv[]){

  // make sure we have enough commandline arguments
  if (argc < 3){
    std::cout << "use parameters: filename1, filename2, (filenametruth)" << std::endl;;
    std::exit(1);
  }

  // get the filenames
  std::string filename1 (argv[1]);
  std::string filename2 (argv[2]);
  std::string truthfilename;
  if (argc > 3){
    truthfilename = argv[3];
  } else {
    truthfilename = "";
  }

  // load the images, and make sure the exist
  cv::Mat image1 = cv::imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
  if (image1.empty()){
    std::cout << "image 1 not found" << std::endl;
    std::exit(1);
  }

  cv::Mat image2 = cv::imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
  if (image2.empty()){
    std::cout << "image2 not found" << std::endl;
    std::exit(1);
  }

  cv::Mat_<cv::Vec2d> truth;
  // load truthfile
  if (truthfilename != ""){
    truth = loadBarronFile(truthfilename);
  }

  // create window
  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // parameters for computation
  int keyCode = 0;
  std::unordered_map<std::string, parameter> parameters;
  setupParameters(parameters);


  // create flowfield matrix using Mat_ to get () access
  cv::Mat_<cv::Vec2d> flowfield(image1.size());

  // create image for display and helper matrixes
  cv::Mat displayimage(image1.size().height, image1.size().width + flowfield.size().width, CV_8UC3);

  // create trackbars for every parameter
  for (auto &i: parameters){
    //parameter *p = new parameter(i.second);
    cv::createTrackbar(i.first, WINDOW_NAME, &i.second.value, i.second.maxvalue, TrackbarCallback, static_cast<void*>(&i.second));
  }

  // main loop which recomputes the optical flow with the new parameters
  while(true){

    // copy image1 into displayimage
    cv::Mat left(displayimage, cv::Rect(0, 0, image1.size().width, image1.size().height));
    cv::Mat right(displayimage, cv::Rect(image1.size().width, 0, flowfield.size().width, flowfield.size().height));
    cv::cvtColor(image1, left, CV_GRAY2RGB);
    computeColorFlowField2(flowfield).copyTo(right);
    std::cout << "AAE: " << CalcAngularError(flowfield, truth) << std::endl;

    cv::imshow(WINDOW_NAME, displayimage);

    keyCode = cv::waitKey();
    // quit on ESC Key
    if (keyCode == 27){
      std::exit(0);
    }

    // recompute on Enter Key
    if (keyCode == 13){
      flowfield = computeFlowField(image1, image2, parameters);

      std::cout << std::endl << "recomputed flow with:" << std::endl;
      for (auto i: parameters){
        std::cout << i.first << ": " << std::floor((double)i.second.value*100/i.second.divfactor)/100 << std::endl;
      }
    }
  }
}
