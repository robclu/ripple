//==--- ripple/tests/storage/storage_traits_tests.hpp --------*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits_tests.hpp
/// \brief This file contains tests for storage traits.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP
#define RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP

#include <ripple/utility/number.hpp>
#include <ripple/storage/storage_traits.hpp>

TEST(storage_storage_traits, can_determine_layout_types) {
  struct Test {};

  auto b1 = ripple::is_storage_layout_v<ripple::strided_layout_t>;
  auto b2 = ripple::is_storage_layout_v<ripple::contiguous_layout_t>;
  auto b3 = ripple::is_storage_layout_v<int>;
  auto b4 = ripple::is_storage_layout_v<Test>;

  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
  EXPECT_FALSE(b4);
}

// Usual struct:
struct Normal {};
// Struct with multiple types.
template <typename... Ts>
struct Params {};
// Struct with a default type.
template <typename T, typename U = ripple::contiguous_layout_t>
struct Default {};
// Struct with int.
template <int I, typename... Ts>
struct WithInt {};

TEST(storage_storage_traits, can_determine_if_type_has_storage_layout) {
  using mult_2_t   = Params<int, ripple::contiguous_layout_t>;
  using mult_3_t   = Params<ripple::strided_layout_t, int, float>;
  using default_t  = Default<int>;
  using with_int_t = WithInt<4, float, ripple::strided_layout_t>;
  using int_wrap_t = Params<ripple::Int64<4>, float, ripple::strided_layout_t>;

  auto b1 = ripple::has_storage_layout_v<Normal>;
  auto b2 = ripple::has_storage_layout_v<mult_2_t>;
  auto b3 = ripple::has_storage_layout_v<mult_3_t>;
  auto b4 = ripple::has_storage_layout_v<default_t>;
  auto b5 = ripple::has_storage_layout_v<with_int_t>;
  auto b6 = ripple::has_storage_layout_v<int_wrap_t>;

  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);
  EXPECT_TRUE(b3);
  EXPECT_TRUE(b4);
  EXPECT_FALSE(b5);
  EXPECT_TRUE(b6);
}

#endif // RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP
