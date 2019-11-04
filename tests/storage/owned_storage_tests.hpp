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

TEST(storage_owned_storage, can_get_number_of_components_for_types) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::OwnedStorage<e_3i_t, e_2f_t, int>;

  storage_t s;
  constexpr auto size_i3 = s.components_of<0>();
  constexpr auto size_f2 = s.components_of<1>();
  constexpr auto size_i  = s.components_of<2>();

  EXPECT_EQ(size_i3, std::size_t{3});
  EXPECT_EQ(size_f2, std::size_t{2});
  EXPECT_EQ(size_i , std::size_t{1});
}

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

//==--- [padding] ----------------------------------------------------------==//

TEST(storage_owned_storage, correct_size_with_padding) {
  using elem_2i_t = ripple::StorageElement<int, 4>;
  using storage_t = ripple::OwnedStorage<bool, elem_2i_t, float>;
  storage_t s;

  // Padding required for alignment of ints after bool:
  EXPECT_EQ(sizeof(s), sizeof(bool) + 3 + sizeof(int) * 4 + sizeof(float));
}

//==--- [indexing] ---------------------------------------------------------==//

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

TEST(storage_owned_storage, can_access_indexable_types_at_runtime) {
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

  EXPECT_EQ((s.get<0, 0>()), s.get<0>(0));
  EXPECT_EQ((s.get<0, 1>()), s.get<0>(1));
  EXPECT_EQ((s.get<0, 2>()), s.get<0>(2));
  EXPECT_EQ((s.get<0, 3>()), s.get<0>(3));
}

#endif // RIPPLE_TESTS_OWNED_STORAGE_TESTS_HPP

