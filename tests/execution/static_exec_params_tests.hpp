//==--- ../execution/static_exec_params_tests.cpp ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  static_exec_params_tests.hpp
/// \brief This file contains tests for a static execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_EXECUTION_STATIC_EXECUTION_PARAMS_TESTS_HPP
#define RIPPLE_TESTS_EXECUTION_STATIC_EXECUTION_PARAMS_TESTS_HPP

#include <ripple/execution/static_execution_params.hpp>
#include <gtest/gtest.h>

TEST(execution_static_exec_params, can_create_single_grain_no_shared) {
  constexpr std::size_t size_x = 1024, size_y = 1, size_z = 1;
  using exec_params_t = ripple::StaticExecParams<size_x, size_y, size_z>;

  exec_params_t params;

  constexpr auto size_1d = params.template size<1>();
  constexpr auto size_2d = params.template size<2>();
  constexpr auto size_3d = params.template size<3>();
  EXPECT_EQ(size_1d, size_x);
  EXPECT_EQ(size_2d, size_x * size_y);
  EXPECT_EQ(size_3d, size_x * size_y * size_z);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.is_fixed(), true);
  EXPECT_EQ(params.padding() , std::size_t{0});
}

TEST(execution_static_exec_params, can_create_simple_multidim) {
  constexpr std::size_t size_x = 16, size_y = 16, size_z = 4;
  using exec_params_t = ripple::StaticExecParams<size_x, size_y, size_z>;

  exec_params_t params;

  constexpr auto size_1d = params.template size<1>();
  constexpr auto size_2d = params.template size<2>();
  constexpr auto size_3d = params.template size<3>();
  EXPECT_EQ(size_1d, size_x);
  EXPECT_EQ(size_2d, size_x * size_y);
  EXPECT_EQ(size_3d, size_x * size_y * size_z);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.is_fixed(), true);
  EXPECT_EQ(params.padding() , std::size_t{0});
}

TEST(execution_static_exec_params, can_create_padded_multidim) {
  constexpr std::size_t 
    size_x     = 16,
    size_y     = 16,
    size_z     = 4 ,
    grain_size = 1 ,
    padding    = 3 ;
  using exec_params_t =
    ripple::StaticExecParams<size_x, size_y, size_z, grain_size, padding>;

  exec_params_t params;

  constexpr auto sxp     = size_x + 2 * padding;
  constexpr auto syp     = size_y + 2 * padding;
  constexpr auto szp     = size_z + 2 * padding;
  constexpr auto size_1d = params.template size<1>();
  constexpr auto size_2d = params.template size<2>();
  constexpr auto size_3d = params.template size<3>();
  EXPECT_EQ(size_1d, sxp);
  EXPECT_EQ(size_2d, sxp * syp);
  EXPECT_EQ(size_3d, sxp * syp * szp);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.is_fixed()  , true);
  EXPECT_EQ(params.padding()   , padding);
  EXPECT_EQ(params.grain_size(), grain_size);
}

TEST(execution_static_exec_params, can_create_mutligrain) {
  constexpr std::size_t 
    size_x     = 16,
    size_y     = 16,
    size_z     = 4 ,
    grain_size = 3 ,
    padding    = 3 ;
  using exec_params_t =
    ripple::StaticExecParams<size_x, size_y, size_z, grain_size, padding>;

  exec_params_t params;

  constexpr auto sxp     = size_x + 2 * padding;
  constexpr auto syp     = size_y + 2 * padding;
  constexpr auto szp     = size_z + 2 * padding;
  constexpr auto size_1d = params.template size<1>();
  constexpr auto size_2d = params.template size<2>();
  constexpr auto size_3d = params.template size<3>();
  EXPECT_EQ(size_1d, sxp);
  EXPECT_EQ(size_2d, sxp * syp);
  EXPECT_EQ(size_3d, sxp * syp * szp);

  EXPECT_EQ(params.size(ripple::dim_x), size_x);
  EXPECT_EQ(params.size(ripple::dim_y), size_y);
  EXPECT_EQ(params.size(ripple::dim_z), size_z);

  EXPECT_EQ(params.is_fixed()  , true);
  EXPECT_EQ(params.padding()   , padding);
  EXPECT_EQ(params.grain_size(), grain_size);
}

#endif // RIPPLE_TESTS_EXECUTION_STATIC_EXECUTION_PARAMS_TESTS_HPP

