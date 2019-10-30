//==--- ripple/tests/storage/strided_storage_tests.hpp -------*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  strided_storage_tests.hpp
/// \brief This file contains tests for strided storage.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STRIDED_STORAGE_TESTS_HPP
#define RIPPLE_TESTS_STRIDED_STORAGE_TESTS_HPP

#include <ripple/storage/strided_storage.hpp>

TEST(storage_strided_storage, dyn_allocation_size_simple_types) {
  using storage_t   = ripple::StridedStorage<int, float>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  EXPECT_EQ(
    allocator_t::allocation_size(elements), 
    (sizeof(int) + sizeof(float)) * elements
  );
}

TEST(storage_strided_storage, dyn_allocation_size_storage_element_types) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the elem_2f elements will require
  // 2 * sizeof(float) bytes, and the elem_3i elements will require
  // 3 * sizeof(int) bytes.
  EXPECT_EQ(
    allocator_t::allocation_size(elements), 
    (3 * sizeof(int) + 2 * sizeof(float)) * elements
  );
}

TEST(storage_strided_storage, dyn_allocation_size_mixed_element_types) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_b_t    = bool;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_b_t, elem_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the elem_2f elements will require
  // 2 * sizeof(float) bytes, and the elem_3i elements will require
  // 3 * sizeof(int) bytes, while the elem_b requires sizeof(bool).
  EXPECT_EQ(
    allocator_t::allocation_size(elements),
    (3 * sizeof(int) + 2 * sizeof(float) + sizeof(bool)) * elements
  );
}

TEST(storage_strided_storage, static_allocation_size_simple_types) {
  using storage_t   = ripple::StridedStorage<int, float>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  constexpr auto size     = allocator_t::allocation_size<elements>();
  EXPECT_EQ(size, (sizeof(int) + sizeof(float)) * elements);
}

TEST(storage_strided_storage, static_allocation_size_storage_element_types) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the elem_2f elements will require
  // 2 * sizeof(float) bytes, and the elem_3i elements will require
  // 3 * sizeof(int) bytes.
  constexpr auto size = allocator_t::allocation_size<elements>();
  EXPECT_EQ(size, (3 * sizeof(int) + 2 * sizeof(float)) * elements);
}

TEST(storage_strided_storage, static_allocation_size_mixed_element_types) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_b_t    = bool;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_b_t, elem_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the elem_2f elements will require
  // 2 * sizeof(float) bytes, and the elem_3i elements will require
  // 3 * sizeof(int) bytes, while the elem_b requires sizeof(bool).
  constexpr auto size = allocator_t::allocation_size<elements>();
  EXPECT_EQ(
    size, 
    (3 * sizeof(int) + 2 * sizeof(float) + sizeof(bool)) * elements
  );
}

TEST(storage_strided_storage, can_create_and_access_simple_types_1d) {
  using storage_t   = ripple::StridedStorage<int, float>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<1>;


  constexpr auto elements = 100;
  constexpr auto size     = allocator_t::allocation_size(elements);
  auto           space    = space_t{elements};
  void*          data     = malloc(size);

  auto storage = allocator_t::create(data, space);

  // Separate loops are needed here to make sure that the offsetting is correct
  // and that none of the data is overwritten by a different offset.
  for (const auto i : ripple::range(elements)) {
    auto s = allocator_t::offset(storage, space, i);
    s.get<0>() = i;
    s.get<1>() = static_cast<float>(i);
  }

  for (const auto i : ripple::range(elements)) {
    const auto s = allocator_t::offset(storage, space, i);
    EXPECT_EQ(s.get<0>(), i);
    EXPECT_EQ(s.get<1>(), static_cast<float>(i));
  }  
  
  free(data);
}

TEST(storage_strided_storage, can_create_and_access_simple_elements_2d) {
  using storage_t   = ripple::StridedStorage<int, float, bool>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<2>;


  constexpr auto elements_x = 12;
  constexpr auto elements_y = 23;
  auto           space    = space_t{elements_x, elements_y};
  auto           size     = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) + sizeof(float) + sizeof(bool)) * elements_x * elements_y
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      auto s = allocator_t::offset(storage, space, i, j);
   
      // elem_3i 
      s.get<0>() = i;
      s.get<1>() = static_cast<float>(i + 1);
      s.get<2>() = (i % 2 == 0) ? true : false;

      EXPECT_EQ(s.get<0>(), i);
      EXPECT_EQ(s.get<1>(), static_cast<float>(i + 1));
      EXPECT_EQ(s.get<2>(), ((i % 2 == 0) ? true : false));
    }
  } 

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      const auto s = allocator_t::offset(storage, space, i, j);
      EXPECT_EQ(s.get<0>(), i);
      EXPECT_EQ(s.get<1>(), static_cast<float>(i + 1));
      EXPECT_EQ(s.get<2>(), ((i % 2 == 0) ? true : false)); 
    }
  } 
  
  free(data);
}


TEST(storage_strided_storage, can_create_and_access_storage_elements_2d) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using elem_b_t    = bool;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_2f_t, elem_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<2>;


  constexpr auto elements_x = 10;
  constexpr auto elements_y = 17;
  auto           space    = space_t{elements_x, elements_y};
  auto           size     = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      auto s = allocator_t::offset(storage, space, i, j);
   
      // elem_3i 
      s.get<0, 0>() = i;
      s.get<0, 1>() = i + 1;
      s.get<0, 2>() = i + 2;
      // elem_2f
      s.get<1, 0>() = static_cast<float>(i);
      s.get<1, 1>() = static_cast<float>(i + 1);
      // elem_b
      s.get<2>() = (i % 2 == 0) ? true : false;
    }
  } 

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      const auto s = allocator_t::offset(storage, space, i, j);
   
      // elem_3i 
      auto e = s.get<0, 0>();
      EXPECT_EQ(e, i);
      EXPECT_EQ((s.get<0, 1>()), i + 1);
      EXPECT_EQ((s.get<0, 2>()), i + 2);
      // elem_3f
      EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
      EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
      // elem_b
      EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);
    }
  } 
  
  free(data);
}

TEST(storage_strided_storage, can_create_and_access_simple_elements_3d) {
  using storage_t   = ripple::StridedStorage<int, float, bool>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;


  constexpr auto elements_x = 21;
  constexpr auto elements_y = 122;
  constexpr auto elements_z = 31;
  auto           space      = space_t{elements_x, elements_y, elements_z};
  auto           size       = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) + sizeof(float) + sizeof(bool)) * 
    elements_x * elements_y * elements_z
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
   
        // elem_3i 
        s.get<0>() = i + j + k;
        s.get<1>() = static_cast<float>(i + j + 1);
        s.get<2>() = (i % 2 == 0) ? true : false;

        EXPECT_EQ(s.get<0>(), i + j + k);
        EXPECT_EQ(s.get<1>(), static_cast<float>(i + j + 1));
        EXPECT_EQ(s.get<2>(), ((i % 2 == 0) ? true : false));
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        const auto s = allocator_t::offset(storage, space, i, j, k);
        EXPECT_EQ(s.get<0>(), i + j + k);
        EXPECT_EQ(s.get<1>(), static_cast<float>(i + j + 1));
        EXPECT_EQ(s.get<2>(), ((i % 2 == 0) ? true : false)); 
      }
    }
  } 
  
  free(data);
}


TEST(storage_strided_storage, can_create_and_access_storage_elements_3d) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using elem_b_t    = bool;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_2f_t, elem_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;


  constexpr auto elements_x = 13;
  constexpr auto elements_y = 107;
  constexpr auto elements_z = 19;
  auto           space    = space_t{elements_x, elements_y, elements_z};
  auto           size     = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y * elements_z
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
   
        // elem_3i 
        s.get<0, 0>() = i;
        s.get<0, 1>() = i + 1;
        s.get<0, 2>() = i + 2;
        // elem_2f
        s.get<1, 0>() = static_cast<float>(i);
        s.get<1, 1>() = static_cast<float>(i + 1);
        // elem_b
        s.get<2>() = (i % 2 == 0) ? true : false;
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        const auto s = allocator_t::offset(storage, space, i, j, k);
   
        // elem_3i 
        auto e = s.get<0, 0>();
        EXPECT_EQ(e, i);
        EXPECT_EQ((s.get<0, 1>()), i + 1);
        EXPECT_EQ((s.get<0, 2>()), i + 2);
        // elem_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
        // elem_b
        EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

TEST(storage_strided_storage, can_copy_and_move_construct_elements) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using elem_b_t    = bool;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_2f_t, elem_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;

  constexpr auto elements_x = 13;
  constexpr auto elements_y = 107;
  constexpr auto elements_z = 19;
  auto           space    = space_t{elements_x, elements_y, elements_z};
  auto           size     = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y * elements_z
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

 for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
   
        // elem_3i 
        s.get<0, 0>() = i;
        s.get<0, 1>() = i + 1;
        s.get<0, 2>() = i + 2;
        // elem_2f
        s.get<1, 0>() = static_cast<float>(i);
        s.get<1, 1>() = static_cast<float>(i + 1);
        // elem_b
        s.get<2>() = (i % 2 == 0) ? true : false;
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto a = allocator_t::offset(storage, space, i, j, k);
  
        // Test copy:
        auto s(a);
   
        // elem_3i 
        EXPECT_EQ((s.get<0, 0>()), i);
        EXPECT_EQ((s.get<0, 1>()), i + 1);
        EXPECT_EQ((s.get<0, 2>()), i + 2);
        // elem_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
        // elem_b
        EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);

        // Test move:
        auto s1(std::move(s));

        // elem_3i 
        EXPECT_EQ((s1.get<0, 0>()), i);
        EXPECT_EQ((s1.get<0, 1>()), i + 1);
        EXPECT_EQ((s1.get<0, 2>()), i + 2);
        // elem_3f
        EXPECT_EQ((s1.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s1.get<1, 1>()), static_cast<float>(i + 1));
        // elem_b
        EXPECT_EQ(s1.get<2>(), (i % 2 == 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

TEST(storage_strided_storage, can_copy_and_move_assign_elements) {
  using elem_3i_t   = ripple::StorageElement<int, 3>;
  using elem_2f_t   = ripple::StorageElement<float, 2>;
  using elem_b_t    = bool;
  using storage_t   = ripple::StridedStorage<elem_3i_t, elem_2f_t, elem_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;

  constexpr auto elements_x = 13;
  constexpr auto elements_y = 107;
  constexpr auto elements_z = 19;
  auto           space    = space_t{elements_x, elements_y, elements_z};
  auto           size     = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y * elements_z
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

 for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto s = allocator_t::offset(storage, space, i, j, k);
   
        // elem_3i 
        s.get<0, 0>() = i;
        s.get<0, 1>() = i + 1;
        s.get<0, 2>() = i + 2;
        // elem_2f
        s.get<1, 0>() = static_cast<float>(i);
        s.get<1, 1>() = static_cast<float>(i + 1);
        // elem_b
        s.get<2>() = (i % 2 == 0) ? true : false;
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto a = allocator_t::offset(storage, space, i, j, k);
  
        // Test copy:
        auto s = a;
   
        // elem_3i 
        EXPECT_EQ((s.get<0, 0>()), i);
        EXPECT_EQ((s.get<0, 1>()), i + 1);
        EXPECT_EQ((s.get<0, 2>()), i + 2);
        // elem_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
        // elem_b
        EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);

        // Test move:
        auto s1 = std::move(s);

        // elem_3i 
        EXPECT_EQ((s1.get<0, 0>()), i);
        EXPECT_EQ((s1.get<0, 1>()), i + 1);
        EXPECT_EQ((s1.get<0, 2>()), i + 2);
        // elem_3f
        EXPECT_EQ((s1.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s1.get<1, 1>()), static_cast<float>(i + 1));
        // elem_b
        EXPECT_EQ(s1.get<2>(), (i % 2 == 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

#endif // RIPPLE_TESTS_STRIDED_STORAGE_TESTS_HPP
