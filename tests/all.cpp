//==--- ripple/core/tests/all.cpp ------------------------------- -*- C++ -*- ---==//
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
#include "boundary/boundary_tests.hpp"
#include "container/container_tests.hpp"
#include "execution/execution_tests.hpp"
#include "iterator/iterator_tests.hpp"
#include "math/math_tests.hpp"
#include "multidim/multidim_tests.hpp"
#include "storage/storage_tests.hpp"
#include "utility/utility_tests.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
