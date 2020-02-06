//==--- ripple/core/tests/iterator.cpp -------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  iterator.cpp
/// \brief This file contains the main function for iterator tests.
//
//==------------------------------------------------------------------------==//

#include "iterator/iterator_tests.hpp"
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
