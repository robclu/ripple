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

#endif // RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP
