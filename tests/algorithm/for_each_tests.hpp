//==--- ripple/tests/algorithm_tests.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  for_each_tests.hpp
/// \brief This file includes tests for for_each functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_ALGORITHM_FOR_EACH_TESTS_HPP
#define RIPPLE_TESTS_ALGORITHM_FOR_EACH_TESTS_HPP

#include <ripple/core/algorithm/for_each.hpp>
#include <gtest/gtest.h>

struct TestType { float x = 0.0; float y = 0.0; };

TEST(algorithm_for_each, can_use_for_each_on_non_const_tuple) {
  auto tuple = ripple::make_tuple(3, 4.4, TestType{3.0, 4.0});
  ripple::for_each(tuple, [] (auto& e) -> void {
    if constexpr (std::is_same_v<TestType, std::decay_t<decltype(e)>>) {
      e.x = 2.0;
      e.y = 3.0;
    } else {
      e = 4;
    }
  });

  EXPECT_EQ(ripple::get<0>(tuple)  , 4  );
  EXPECT_EQ(ripple::get<1>(tuple)  , 4.0);
  EXPECT_EQ(ripple::get<2>(tuple).x, 2.0);
  EXPECT_EQ(ripple::get<2>(tuple).y, 3.0);
}

TEST(algorithm_for_each, can_use_for_each_on_const_tuple) {
  const auto tuple = ripple::make_tuple(3, 4.4, TestType{3.0, 4.0});
  int    x = 0;
  double y = 0;
  TestType t;
  ripple::for_each(tuple, [&] (const auto& e) -> void {
    using e_t = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<TestType, e_t>) {
      t.x = e.x;
      t.y = e.y;
    }
    if constexpr (std::is_same_v<int, e_t>)  {
      x = e;
    }
    if constexpr (std::is_same_v<double, e_t>)  {
      y = e;
    }
  });

  EXPECT_EQ(ripple::get<0>(tuple)  , x  );
  EXPECT_EQ(ripple::get<1>(tuple)  , y  );
  EXPECT_EQ(ripple::get<2>(tuple).x, t.x);
  EXPECT_EQ(ripple::get<2>(tuple).y, t.y);
}

TEST(algorithm_for_each, can_use_for_each_on_rvalue_tuple) {
  int    x = 0;
  double y = 0;
  TestType t;
  ripple::for_each(ripple::make_tuple(3, 4.4, TestType{3.0, 4.0}), 
    [&] (auto&& e) -> void {
      using e_t = std::decay_t<decltype(e)>;
      if constexpr (std::is_same_v<TestType, e_t>) {
        t.x = e.x;
        t.y = e.y;
      }
      if constexpr (std::is_same_v<int, e_t>)  {
        x = e;
      }
      if constexpr (std::is_same_v<double, e_t>)  {
        y = e;
      }
    }
  );

  EXPECT_EQ(x  , 3  );
  EXPECT_EQ(y  , 4.4);
  EXPECT_EQ(t.x, 3.0);
  EXPECT_EQ(t.y, 4.0);
}

TEST(algorithm_for_each, can_use_for_each_on_parameter_pack) {
  int      x = 0;
  double   y = 0;
  int      z = 3;
  TestType t;
  ripple::for_each(
    [&] (auto&& e) -> void {
      using e_t = std::decay_t<decltype(e)>;
      if constexpr (std::is_same_v<TestType, e_t>) {
        t.x = e.x;
        t.y = e.y;
      }
      if constexpr (std::is_same_v<int, e_t>)  {
        x = e;
      }
      if constexpr (std::is_same_v<double, e_t>)  {
        y = e;
      }
    }, z, 4.4, TestType{3.0, 4.0}
  );

  EXPECT_EQ(x  , 3  );
  EXPECT_EQ(z  , 3  );
  EXPECT_EQ(y  , 4.4);
  EXPECT_EQ(t.x, 3.0);
  EXPECT_EQ(t.y, 4.0);
}


#endif // RIPPLE_TESTS_ALGORITHM_FOR_EACH_TESTS_HPP