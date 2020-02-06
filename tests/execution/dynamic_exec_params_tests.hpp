//==--- ../execution/dynamic_exec_params_tests.cpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dynamic_exec_params_tests.hpp
/// \brief This file contains tests for a dynamic execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_EXECUTION_DYNAMIC_EXECUTION_PARAMS_TESTS_HPP
#define RIPPLE_TESTS_EXECUTION_DYNAMIC_EXECUTION_PARAMS_TESTS_HPP

#include <ripple/core/execution/dynamic_execution_params.hpp>
#include <gtest/gtest.h>

TEST(execution_dynamic_exec_params, can_create_simple_multidim) {
  using exec_params_t = ripple::DynamicExecParams<ripple::VoidShared>;

  std::size_t size_x = 16, size_y = 16, size_z = 4;
  exec_params_t params(size_x, size_y, size_z);

  EXPECT_EQ(params.template size<1>(), size_x);
  EXPECT_EQ(params.template size<2>(), size_x * size_y);
  EXPECT_EQ(params.template size<3>(), size_x * size_y * size_z);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.padding(), std::size_t{0});
  using traits_t = ripple::ExecTraits<exec_params_t>;
  EXPECT_EQ(traits_t::is_static  , false);
  EXPECT_EQ(traits_t::uses_shared, false);
}

TEST(execution_dynamic_exec_params, can_create_padded_multidim) {
  using exec_params_t = ripple::DynamicExecParams<ripple::VoidShared>;

  std::size_t size_x = 16, size_y = 16, size_z = 4, padding = 2;
  exec_params_t params(padding, size_x, size_y, size_z);

  const auto sxp = size_x + 2 * padding;
  const auto syp = size_y + 2 * padding;
  const auto szp = size_z + 2 * padding;
  EXPECT_EQ(params.template size<1>(), sxp);
  EXPECT_EQ(params.template size<2>(), sxp * syp);
  EXPECT_EQ(params.template size<3>(), sxp * syp * szp);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.padding(), padding);
  using traits_t = ripple::ExecTraits<exec_params_t>;
  EXPECT_EQ(traits_t::is_static  , false);
  EXPECT_EQ(traits_t::uses_shared, false);
}

TEST(execution_dynamic_exec_params, can_detect_shared_usage) {
  using exec_params_t = ripple::DynamicExecParams<float>;

  std::size_t size_x = 16, size_y = 16, size_z = 4, padding = 2;
  exec_params_t params(padding, size_x, size_y, size_z);

  const auto sxp = size_x + 2 * padding;
  const auto syp = size_y + 2 * padding;
  const auto szp = size_z + 2 * padding;
  EXPECT_EQ(params.template size<1>(), sxp);
  EXPECT_EQ(params.template size<2>(), sxp * syp);
  EXPECT_EQ(params.template size<3>(), sxp * syp * szp);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.padding(), padding);
  using traits_t = ripple::ExecTraits<exec_params_t>;
  EXPECT_EQ(traits_t::is_static  , false);
  EXPECT_EQ(traits_t::uses_shared, true);
}

#endif // RIPPLE_TESTS_EXECUTION_DYNAMIC_EXECUTION_PARAMS_TESTS_HPP

