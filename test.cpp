#include <iostream>
#include <vector>
#include "image_class.h"
#include "filehandling.h"

int main(){
  FlowField truth;

  loadBarronFile("yos_truth.F", truth);
  truth.writeToPNG("flowfield_truth.png");

  return 0;
}
