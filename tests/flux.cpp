//==--- ripple/tests/flux.cpp ------------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 20120 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux.cpp
/// \brief This file contains the main function for flux tests.
//
//==------------------------------------------------------------------------==//

#include "flux/flux_tests.hpp"
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
