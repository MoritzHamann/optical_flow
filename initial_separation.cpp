#include "initial_separation.hpp"


void initial_segmentation(const cv::Mat_<cv::Vec2d> &flowfield,
                          //const cv::Mat_<cv::Vec2d> &initialflowfield
                          cv::Mat_<double> &phi,
                          const std::unordered_map<std::string, parameter> &parameters
                        ){

  //segementFlowfield(flowfield, phi, parameters);

  for (int i = 0; i < phi.rows; i++){
    for (int j = 0; j < phi.cols; j++){
      if (flowfield(i,j)[0] > 1 && flowfield(i,j)[1] < 0.5){
        phi(i,j) = 1;
      } else {
        phi(i,j) = -1;
      }
    }
  }

  //phi = 0;
  
}


void segementFlowfield(const cv::Mat_<cv::Vec2d> &f, cv::Mat_<double> &phi, const std::unordered_map<std::string, parameter> &parameters){

  // helper variables
  int blocksize = parameters.at("blocksize_sep").value;
  int numx = std::ceil((float)f.cols/blocksize), numy = std::ceil((float)f.rows/blocksize);
  cv::Mat_<cv::Vec6d> affine(numy, numx), affine_tmp;
  cv::Mat_<bool> valid(numy, numx);
  cv::Mat_<double> coords(blocksize*blocksize, 3), flow_block(blocksize*blocksize, 2);


  double xmax, ymax;
  // estimated affine parameters for each block
  for (int i = 0; i < numy; i++){
    for (int j = 0; j < numx; j++){
      xmax = ((j+1) * blocksize < f.cols) ? (j+1) * blocksize : f.cols;
      ymax = ((i+1) * blocksize < f.rows) ? (i+1) * blocksize : f.rows;

      int i = 0;  // easiest solution TODO: change later maybe
      for (int y = i*blocksize; y < ymax; y++){
        for (int x = j*blocksize; x < xmax; x++){
          flow_block[i][0] = f(y,x)[0];
          flow_block[i][1] = f(y,x)[1];
          coords[i][0] = x;
          coords[i][1] = y;
          coords[i][2] = 1;
          i++;
        }
      }

      cv::solve(coords, flow_block, affine_tmp, cv::DECOMP_QR);
      affine(i,j) = affine_tmp.clone();


    }
  }

  // only use "good" blocks

  //

}
