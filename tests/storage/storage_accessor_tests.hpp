//==--- ripple/core/tests/storage/storage_accessor_tests.hpp ------*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_accessor_tests.hpp
/// \brief This file contains tests for storage accessing.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STORAGE_ACCESSOR_TESTS_HPP
#define RIPPLE_TESTS_STORAGE_ACCESSOR_TESTS_HPP

#include <ripple/core/storage/owned_storage.hpp>
#include <ripple/core/storage/strided_storage_view.hpp>
#include <ripple/core/storage/contiguous_storage_view.hpp>

TEST(storage_storage_access, can_assign_between_strided_and_owned) {
  using e_3i_t        = ripple::StorageElement<int, 3>;
  using e_2f_t        = ripple::StorageElement<float, 2>;
  using e_b_t         = bool;
  using storage_t     = ripple::StridedStorageView<e_3i_t, e_2f_t, e_b_t>;
  using own_storage_t = ripple::OwnedStorage<e_3i_t, e_2f_t, e_b_t>;
  using allocator_t   = typename storage_t::allocator_t;
  using space_t       = ripple::DynamicMultidimSpace<3>;

  constexpr auto elements_x = 13;
  constexpr auto elements_y = 107;
  constexpr auto elements_z = 19;
  const auto     space      = space_t{elements_x, elements_y, elements_z};
  const auto     size       = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y * elements_z
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  own_storage_t o;
  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
   
        // e_3i 
        s.get<0, 0>() = i;
        s.get<0, 1>() = i + 1;
        s.get<0, 2>() = i + 2;
        // e_2f
        s.get<1, 0>() = static_cast<float>(i);
        s.get<1, 1>() = static_cast<float>(i + 1);
        // e_b
        s.get<2>() = (i % 2 == 0) ? true : false;

        // Copy from strided:
        o = s;

        // Check copied correctly:
        EXPECT_EQ((o.get<0, 0>()), i);
        EXPECT_EQ((o.get<0, 1>()), i + 1);
        EXPECT_EQ((o.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((o.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((o.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(o.get<2>(), (i % 2 == 0) ? true : false);

        // modify o:
        o.get<0, 0>() = i + 10;
        o.get<0, 1>() = i + 20;
        o.get<0, 2>() = i + 30;
        // e_2f
        o.get<1, 0>() = static_cast<float>(i + 10);
        o.get<1, 1>() = static_cast<float>(i + 20);
        // e_b
        o.get<2>() = (i % 2 != 0) ? true : false;

        // Set s again:
        s = o;
      }
    }
  } 

  // Check that strided storage was not changed:
  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
  
        // e_3i 
        EXPECT_EQ((s.get<0, 0>()), i + 10);
        EXPECT_EQ((s.get<0, 1>()), i + 20);
        EXPECT_EQ((s.get<0, 2>()), i + 30);
        // e_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i + 10));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 20));
        // e_b
        EXPECT_EQ(s.get<2>(), (i % 2 != 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

TEST(storage_storage_access, can_assign_between_contiguous_and_owned) {
  using e_3i_t        = ripple::StorageElement<int, 3>;
  using e_2f_t        = ripple::StorageElement<float, 2>;
  using e_b_t         = bool;
  using storage_t     = ripple::ContiguousStorageView<e_3i_t, e_2f_t, e_b_t>;
  using own_storage_t = ripple::OwnedStorage<e_3i_t, e_2f_t, e_b_t>;
  using allocator_t   = typename storage_t::allocator_t;
  using space_t       = ripple::DynamicMultidimSpace<3>;

  constexpr auto elements_x = 13;
  constexpr auto elements_y = 107;
  constexpr auto elements_z = 19;
  const auto     space      = space_t{elements_x, elements_y, elements_z};
  const auto     size       = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y * elements_z
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  own_storage_t o;
  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
   
        // e_3i 
        s.get<0, 0>() = i;
        s.get<0, 1>() = i + 1;
        s.get<0, 2>() = i + 2;
        // e_2f
        s.get<1, 0>() = static_cast<float>(i);
        s.get<1, 1>() = static_cast<float>(i + 1);
        // e_b
        s.get<2>() = (i % 2 == 0) ? true : false;

        // Copy from contiguous:
        o = s;

        // Check copied correctly:
        EXPECT_EQ((o.get<0, 0>()), i);
        EXPECT_EQ((o.get<0, 1>()), i + 1);
        EXPECT_EQ((o.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((o.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((o.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(o.get<2>(), (i % 2 == 0) ? true : false);

        // modify o:
        o.get<0, 0>() = i + 10;
        o.get<0, 1>() = i + 20;
        o.get<0, 2>() = i + 30;
        // e_2f
        o.get<1, 0>() = static_cast<float>(i + 10);
        o.get<1, 1>() = static_cast<float>(i + 20);
        // e_b
        o.get<2>() = (i % 2 != 0) ? true : false;

        // Set s again:
        s = o;
      }
    }
  } 

  // Check that strided storage was not changed:
  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
  
        // e_3i 
        EXPECT_EQ((s.get<0, 0>()), i + 10);
        EXPECT_EQ((s.get<0, 1>()), i + 20);
        EXPECT_EQ((s.get<0, 2>()), i + 30);
        // e_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i + 10));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 20));
        // e_b
        EXPECT_EQ(s.get<2>(), (i % 2 != 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

#endif // RIPPLE_TESTS_STORAGE_ACCESSOR_TESTS_HPP

