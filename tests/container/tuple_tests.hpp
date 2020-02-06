//==--- ripple/core/tests/container/tuple_tests.hpp ------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tuple_tests.hpp
/// \brief This file defines tests for tuple functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_TUPLE_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_TUPLE_TESTS_HPP

#include <ripple/core/container/tuple.hpp>
#include <gtest/gtest.h>

//==--- [creation] ---------------------------------------------------------==//

TEST(container_tuple, can_create_default_tuple) {
  ripple::Tuple<float, int, double> t;

  EXPECT_EQ(t.size(), size_t{3});
  EXPECT_EQ(ripple::tuple_traits_t<decltype(t)>::size, size_t{3});
}

TEST(container_tuple, can_access_and_modify_default_tuple_elements) {
  ripple::Tuple<float, int, double> t;

  ripple::get<0>(t) = 3.3f;
  ripple::get<1>(t) = 3;
  ripple::get<2>(t) = 4.7;

  auto  x = ripple::get<0>(t);
  auto& y = ripple::get<0>(t);
  x = 22.3f;

  EXPECT_EQ(ripple::get<0>(t), 3.3f);
  EXPECT_EQ(ripple::get<1>(t), 3   );
  EXPECT_EQ(ripple::get<2>(t), 4.7 );
  EXPECT_EQ(x, 22.3f);
  EXPECT_EQ(y, 3.3f);
}

TEST(container_tuple, can_create_tuple_with_arguments) {
  ripple::Tuple<float, int, double> t(3.3f, 3, 4.7);

  EXPECT_EQ(ripple::get<0>(t), 3.3f);
  EXPECT_EQ(ripple::get<1>(t), 3   );
  EXPECT_EQ(ripple::get<2>(t), 4.7 );

  auto  x = ripple::get<0>(t);
  auto& y = ripple::get<0>(t);
  x = 22.3f;

  EXPECT_EQ(ripple::get<0>(t), 3.3f);
  EXPECT_EQ(ripple::get<1>(t), 3   );
  EXPECT_EQ(ripple::get<2>(t), 4.7 );
  EXPECT_EQ(x, 22.3f);
  EXPECT_EQ(y, 3.3f);
}

TEST(container_tuple, can_copy_construct_tuple) {
  ripple::Tuple<float, int, double> t(3.3f, 3, 4.7);

  EXPECT_EQ(ripple::get<0>(t), 3.3f);
  EXPECT_EQ(ripple::get<1>(t), 3   );
  EXPECT_EQ(ripple::get<2>(t), 4.7 );

  auto tt(t);

  EXPECT_EQ(ripple::get<0>(tt), 3.3f);
  EXPECT_EQ(ripple::get<1>(tt), 3   );
  EXPECT_EQ(ripple::get<2>(tt), 4.7 );
}

TEST(container_tuple, can_move_construct_tuple) {
  ripple::Tuple<float, int, double> t(3.3f, 3, 4.7);

  EXPECT_EQ(ripple::get<0>(t), 3.3f);
  EXPECT_EQ(ripple::get<1>(t), 3   );
  EXPECT_EQ(ripple::get<2>(t), 4.7 );

  auto tt(std::move(t));

  EXPECT_EQ(ripple::get<0>(tt), 3.3f);
  EXPECT_EQ(ripple::get<1>(tt), 3   );
  EXPECT_EQ(ripple::get<2>(tt), 4.7 );
}

TEST(container_tuple, can_create_tuple_with_make_tuple) {
  auto t = ripple::make_tuple(3.3f, 3, 4.7);

  EXPECT_EQ(ripple::get<0>(t), 3.3f);
  EXPECT_EQ(ripple::get<1>(t), 3   );
  EXPECT_EQ(ripple::get<2>(t), 4.7 );

  auto float_match  = std::is_same_v<decltype(ripple::get<0>(t)), float&>;
  auto int_match    = std::is_same_v<decltype(ripple::get<1>(t)), int&>;
  auto double_match = std::is_same_v<decltype(ripple::get<2>(t)), double&>;
  
  EXPECT_TRUE(float_match);
  EXPECT_TRUE(int_match);
  EXPECT_TRUE(double_match);
}

#endif // RIPPLE_TESTS_CONTAINER_TUPLE_TESTS_HPP

