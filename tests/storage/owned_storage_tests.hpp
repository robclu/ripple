//==--- ripple/tests/storage/owned_storage_tests.hpp ---------*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  owned_storage_tests.hpp
/// \brief This file contains tests for owned storage.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_OWNED_STORAGE_TESTS_HPP
#define RIPPLE_TESTS_OWNED_STORAGE_TESTS_HPP

#include <ripple/storage/owned_storage.hpp>

TEST(storage_owned_storage, correct_size_simple_types) {
  using storage_t   = ripple::OwnedStorage<int, float>;
  storage_t s;
  EXPECT_EQ(sizeof(s), sizeof(int) + sizeof(float));
}

TEST(storage_owned_storage, correct_size_storage_element_types) {
  using elem_2i_t = ripple::StorageElement<int, 4>;
  using storage_t = ripple::OwnedStorage<elem_2i_t, float>;
  storage_t s;
  EXPECT_EQ(sizeof(s), sizeof(int) * 4 + sizeof(float));
}

TEST(storage_owned_storage, correct_size_with_padding) {
  using elem_2i_t = ripple::StorageElement<int, 4>;
  using storage_t = ripple::OwnedStorage<bool, elem_2i_t, float>;
  storage_t s;

  // Padding required for alignment of ints after bool:
  EXPECT_EQ(sizeof(s), sizeof(bool) + 3 + sizeof(int) * 4 + sizeof(float));
}

TEST(storage_owned_storage, can_access_and_set_types) {
  using elem_2i_t = ripple::StorageElement<int, 4>;
  using storage_t = ripple::OwnedStorage<elem_2i_t, float>;
  storage_t s;

  s.get<0, 0>() = 1;
  s.get<0, 1>() = 2;
  s.get<0, 2>() = 3;
  s.get<0, 3>() = 4;
  s.get<1>()    = 7.7f;

  EXPECT_EQ((s.get<0, 0>()), 1   );
  EXPECT_EQ((s.get<0, 1>()), 2   );
  EXPECT_EQ((s.get<0, 2>()), 3   );
  EXPECT_EQ((s.get<0, 3>()), 4   );
  EXPECT_EQ((s.get<1>())   , 7.7f);
}

#endif // RIPPLE_TESTS_OWNED_STORAGE_TESTS_HPP

