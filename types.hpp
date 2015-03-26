#ifndef TYPES_HPP
#define TYPES_HPP

#include <string>

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

#endif
