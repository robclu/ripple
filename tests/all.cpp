//==--- ripple/tests/all.cpp -------------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  all.cpp
/// \brief This file contains all tests.
//
//==------------------------------------------------------------------------==//

#include "algorithm/algorithm_tests.hpp"
#include "container/container_tests.hpp"
#include "utility/utility_tests.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
