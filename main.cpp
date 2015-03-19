#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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
  // TODO: load barron file into mat
  // Mat truth = loadBarronFile();

  //

  int pos = 0;
  int poss = 2;
  int keyCode = 0;

  // main loop which recomputes the optical flow with the new parameters

  cv::namedWindow("Optical flow", cv::WINDOW_AUTOSIZE);
  cv::imshow("Optical flow", image1);
  cv::createTrackbar("alpha", "Optical flow", &pos, 1000);
  cv::createTrackbar("maxiter", "Optical flow", &poss, 10);

  //cv::imshow("Optical flow", image2);
  keyCode = cv::waitKey();
  std::cout << keyCode << std::endl;


}

/*int main2(int argc, char *argv[]){

  if (argc < 10){
    std::cout << "use following command line arguments" << std::endl;
    std::cout << "img1 img2 truth numlevel alpha wrapfactor omega maxiter sigma" << std::endl;
    std::exit(1);
  }
  std::string filename1 (argv[1]);
  std::string filename2 (argv[2]);
  std::string truthfilename (argv[3]);
  int level = std::atoi(argv[4]);
  double alpha = std::atof(argv[5]);
  double wrapfactor = std::atof(argv[6]);
  double omega = std::atof(argv[7]);
  int maxiter = std::atoi(argv[8]);
  double sigma = std::atof(argv[9]);

  std::cout << filename1 << std::endl;
  std::cout << filename2 << std::endl;
  std::cout << level << std::endl;
  std::cout << alpha << std::endl;
  std::cout << wrapfactor << std::endl;
  std::cout << omega << std::endl;
  std::cout << maxiter << std::endl;
  std::cout << sigma << std::endl;

  Img image1;
  Img image2;
  FlowField c;
  FlowField d;
  FlowField truth;

  // load files
  loadPNGImage(filename1, image1);
  loadPNGImage(filename2, image2);
  //loadBarronFile(truthfilename, truth);

  // resize flowfields
  c.Resize(image1.Size());
  d.Resize(image2.Size());

  // make sure image 1 and image 2 have same size
  std::pair<int, int> size = image1.Size();
  if (size != image2.Size()){
    std::cout << "Dimension of images are not equal" << std::endl;
    std::exit(1);
  }

  image1.GaussianSmoothOriginal(sigma);
  image2.GaussianSmoothOriginal(sigma);

  HornSchunckLevelLoop(level, maxiter, alpha, omega, wrapfactor, image1, image2, c, d);

  c.writeToPNG("flowfield.png");
  //c.writeErrorToPNG("flowfield-error.png", truth);
  //std::cout << c.CalcAngularError(truth) << std::endl;

}
*/
