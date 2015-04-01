#ifndef TYPES_HPP
#define TYPES_HPP

#include <string>
#include <opencv2/core/core.hpp>

struct parameter {
  std::string name;
  int value;
  int maxvalue;
  int divfactor;
};

struct tensor {
  double J11;
  double J22;
  double J33;
  double J12;
  double J13;
  double J23;
};


template<> class cv::DataType<tensor> {
public:
  typedef tensor channel_type;
  enum {
    channels = 6,
    type=CV_MAKETYPE(64, 6)
  };
};

#endif
