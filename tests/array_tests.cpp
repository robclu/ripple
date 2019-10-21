//==--- cpp/tests/array_tests.cpp -------------------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array_tests.cpp
/// \brief This file contains tests for arrays.
//
//==------------------------------------------------------------------------==//

#include <streamline/container/vec.hpp>
#include <gtest/gtest.h>

TEST(array, can_create_array_impl) {
  auto v = streamline::Vec<float, 3>();

  EXPECT_TRUE(1 == 1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
