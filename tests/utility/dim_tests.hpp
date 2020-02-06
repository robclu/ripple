//==--- ripple/core/tests/utility/dim_tests.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dim_tests.hpp
/// \brief This file implements tests for compile time dimension functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_UTILITY_DIM_TESTS_HPP
#define RIPPLE_TESTS_UTILITY_DIM_TESTS_HPP

#include <ripple/core/utility/dim.hpp>
#include <gtest/gtest.h>


TEST(utility_dim, can_use_at_runtime_as_size_t) {
  EXPECT_EQ(static_cast<std::size_t>(ripple::dim_x), std::size_t{0});
  EXPECT_EQ(static_cast<std::size_t>(ripple::dim_y), std::size_t{1});
  EXPECT_EQ(static_cast<std::size_t>(ripple::dim_z), std::size_t{2});
}

#endif // RIPPLE_TESTS_UTILITY_DIM_TESTS_HPP
