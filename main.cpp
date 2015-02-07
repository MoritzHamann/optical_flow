#include <iostream>
#include "image_class.h"
#include "flow_utility.h"
#include "filehandling.h"
#include "lodepng.h"
#include <utility>
#include "hornschunck.h"

int main(){

  Img image1;
  Img image2;
  FlowField c;
  FlowField d;
  FlowField truth;

  // load files
  loadPGMImage("yos1.pgm", image1);
  loadPGMImage("yos2.pgm", image2);
  loadBarronFile("yos_truth.F", truth);

  // resize flowfields
  c.Resize(image1.Size());
  d.Resize(image2.Size());

  std::cout << "all files loaded" << std::endl;

  // make sure image 1 and image 2 have same size
  std::pair<int, int> size = image1.Size();
  if (size != image2.Size()){
    std::cout << "Dimension of images are not equal" << std::endl;
    std::exit(1);
  }

  std::cout << "dimension tested" << std::endl;

  image1.GaussianSmoothOriginal(1.516);
  image2.GaussianSmoothOriginal(1.516);

  std::cout << "presmoothed" << std::endl;

  HornSchunckLevelLoop(3, 200, 100, 1.95, 0.5, image1, image2, c, d);

  std::cout << "calculated" << std::endl;

  c.writeToPNG("flowfield.png");
  std::cout << c.CalcAngularError(truth) << std::endl;

}
