//==--- ripple/tests/all_device.cu ------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  all_device.cu
/// \brief This file contains all tests for the device.
//
//==------------------------------------------------------------------------==//

#include "algorithm/algorithm_tests_device.hpp"
#include "container/container_tests_device.hpp"
#include "boundary/boundary_tests_device.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
