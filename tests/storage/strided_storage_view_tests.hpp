//==--- ripple/core/tests/storage/strided_storage_tests.hpp -------*- C++ -*- ---==//
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

#include <ripple/core/storage/strided_storage_view.hpp>
#include <ripple/core/utility/dim.hpp>

TEST(storage_strided_storage_view, can_get_number_of_components_for_types) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t, int>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<1>;

  auto space = space_t{1};
  auto* data = malloc(allocator_t::allocation_size(1));
  auto s     = allocator_t::create(data, space);

  constexpr auto size_i3 = s.components_of<0>();
  constexpr auto size_f2 = s.components_of<1>();
  constexpr auto size_i  = s.components_of<2>();

  EXPECT_EQ(size_i3, std::size_t{3});
  EXPECT_EQ(size_f2, std::size_t{2});
  EXPECT_EQ(size_i , std::size_t{1});

  free(data);
}

TEST(storage_strided_storage_view, dyn_allocation_size_simple_types) {
  using storage_t   = ripple::StridedStorageView<int, float>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  EXPECT_EQ(
    allocator_t::allocation_size(elements), 
    (sizeof(int) + sizeof(float)) * elements
  );
}

TEST(storage_strided_storage_view, dyn_allocation_size_storage_element_types) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the e_2f elements will require
  // 2 * sizeof(float) bytes, and the e_3i elements will require
  // 3 * sizeof(int) bytes.
  EXPECT_EQ(
    allocator_t::allocation_size(elements), 
    (3 * sizeof(int) + 2 * sizeof(float)) * elements
  );
}

TEST(storage_strided_storage_view, dyn_allocation_size_mixed_element_types) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_b_t       = bool;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_b_t, e_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the e_2f elements will require
  // 2 * sizeof(float) bytes, and the e_3i elements will require
  // 3 * sizeof(int) bytes, while the e_b requires sizeof(bool).
  EXPECT_EQ(
    allocator_t::allocation_size(elements),
    (3 * sizeof(int) + 2 * sizeof(float) + sizeof(bool)) * elements
  );
}

TEST(storage_strided_storage_view, static_allocation_size_simple_types) {
  using storage_t   = ripple::StridedStorageView<int, float>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  constexpr auto size     = allocator_t::allocation_size<elements>();
  EXPECT_EQ(size, (sizeof(int) + sizeof(float)) * elements);
}

TEST(storage_strided_storage_view, static_allocation_size_storage_types) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the e_2f elements will require
  // 2 * sizeof(float) bytes, and the e_3i elements will require
  // 3 * sizeof(int) bytes.
  constexpr auto size = allocator_t::allocation_size<elements>();
  EXPECT_EQ(size, (3 * sizeof(int) + 2 * sizeof(float)) * elements);
}

TEST(storage_strided_storage_view, static_allocation_size_mixed_types) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_b_t       = bool;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_b_t, e_2f_t>;
  using allocator_t = typename storage_t::allocator_t;

  constexpr auto elements = 100;
  // Here the storage is strided, so each of the e_2f elements will require
  // 2 * sizeof(float) bytes, and the e_3i elements will require
  // 3 * sizeof(int) bytes, while the e_b requires sizeof(bool).
  constexpr auto size = allocator_t::allocation_size<elements>();
  EXPECT_EQ(
    size, 
    (3 * sizeof(int) + 2 * sizeof(float) + sizeof(bool)) * elements
  );
}

//--- [indexing] -----------------------------------------------------------==//

TEST(storage_strided_storage_view, can_access_indexable_types_at_runtime) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using storage_t   = ripple::StridedStorageView<e_3i_t, float>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<1>;


  constexpr auto elements = 100;
  constexpr auto size     = allocator_t::allocation_size(elements);
  const auto     space    = space_t{elements};
  void*          data     = malloc(size);
  auto           storage  = allocator_t::create(data, space);

  for (const auto i : ripple::range(elements)) {
    auto s = allocator_t::offset(storage, space, i);
    s.get<0, 0>() = i;
    s.get<0, 1>() = i + 1;
    s.get<0, 2>() = i + 2;
    s.get<1>() = static_cast<float>(i);
  }

  for (const auto i : ripple::range(elements)) {
    const auto s = allocator_t::offset(storage, space, i);
    EXPECT_EQ((s.get<0, 0>()), i);
    EXPECT_EQ((s.get<0, 1>()), i + 1);
    EXPECT_EQ((s.get<0, 2>()), i + 2);
    EXPECT_EQ(s.get<1>(), static_cast<float>(i));

    EXPECT_EQ((s.get<0, 0>()), s.get<0>(0));
    EXPECT_EQ((s.get<0, 1>()), s.get<0>(1));
    EXPECT_EQ((s.get<0, 2>()), s.get<0>(2));
  }  
  
  free(data);
}

//==--- [access 1d] --------------------------------------------------------==//

TEST(storage_strided_storage_view, can_create_and_access_simple_types_1d) {
  using storage_t   = ripple::StridedStorageView<int, float>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<1>;


  constexpr auto elements = 100;
  constexpr auto size     = allocator_t::allocation_size(elements);
  const auto     space    = space_t{elements};
  void*          data     = malloc(size);
  auto           storage  = allocator_t::create(data, space);

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

//==--- [access 2d] --------------------------------------------------------==//

TEST(storage_strided_storage_view, can_create_and_access_simple_elements_2d) {
  using storage_t   = ripple::StridedStorageView<int, float, bool>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<2>;


  constexpr auto elements_x = 12;
  constexpr auto elements_y = 23;
  const auto     space      = space_t{elements_x, elements_y};
  const auto     size       = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) + sizeof(float) + sizeof(bool)) * elements_x * elements_y
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      auto s = allocator_t::offset(storage, space, i, j);
   
      // e_3i 
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

TEST(storage_strided_storage_view, can_create_and_access_storage_elements_2d) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using e_b_t       = bool;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t, e_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<2>;


  constexpr auto elements_x = 10;
  constexpr auto elements_y = 17;
  const auto     space      = space_t{elements_x, elements_y};
  const auto     size       = allocator_t::allocation_size(space.size());
  EXPECT_EQ(size, 
    (sizeof(int) * 3 + sizeof(float) * 2 + sizeof(bool)) * 
    elements_x * elements_y
  );
  void* data   = malloc(size);
  auto storage = allocator_t::create(data, space);

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      auto s = allocator_t::offset(storage, space, i, j);
   
      // e_3i 
      s.get<0, 0>() = i;
      s.get<0, 1>() = i + 1;
      s.get<0, 2>() = i + 2;
      // e_2f
      s.get<1, 0>() = static_cast<float>(i);
      s.get<1, 1>() = static_cast<float>(i + 1);
      // e_b
      s.get<2>() = (i % 2 == 0) ? true : false;
    }
  } 

  for (const auto j : ripple::range(elements_y)) {
    for (const auto i : ripple::range(elements_x)) {
      const auto s = allocator_t::offset(storage, space, i, j);
   
      // e_3i 
      EXPECT_EQ((s.get<0, 0>()), i);
      EXPECT_EQ((s.get<0, 1>()), i + 1);
      EXPECT_EQ((s.get<0, 2>()), i + 2);
      // e_3f
      EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
      EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
      // e_b
      EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);
    }
  } 
  
  free(data);
}

TEST(storage_strided_storage_view, can_create_and_access_simple_elements_3d) {
  using storage_t   = ripple::StridedStorageView<int, float, bool>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;

  constexpr auto elements_x = 21;
  constexpr auto elements_y = 122;
  constexpr auto elements_z = 31;
  const auto     space      = space_t{elements_x, elements_y, elements_z};
  const auto     size       = allocator_t::allocation_size(space.size());
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
   
        // e_3i 
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


TEST(storage_strided_storage_view, can_create_and_access_storage_elements_3d) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using e_b_t       = bool;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t, e_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;


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
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        const auto s = allocator_t::offset(storage, space, i, j, k);
   
        // e_3i 
        auto e = s.get<0, 0>();
        EXPECT_EQ(e, i);
        EXPECT_EQ((s.get<0, 1>()), i + 1);
        EXPECT_EQ((s.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

TEST(storage_strided_storage_view, can_copy_and_move_construct_elements) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using e_b_t       = bool;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t, e_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;

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
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto a = allocator_t::offset(storage, space, i, j, k);
  
        // Test copy:
        auto s(a);
   
        // e_3i 
        EXPECT_EQ((s.get<0, 0>()), i);
        EXPECT_EQ((s.get<0, 1>()), i + 1);
        EXPECT_EQ((s.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);

        // Test move:
        auto s1(std::move(s));

        // e_3i 
        EXPECT_EQ((s1.get<0, 0>()), i);
        EXPECT_EQ((s1.get<0, 1>()), i + 1);
        EXPECT_EQ((s1.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((s1.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s1.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(s1.get<2>(), (i % 2 == 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

TEST(storage_strided_storage_view, can_copy_and_move_assign_elements) {
  using e_3i_t      = ripple::StorageElement<int, 3>;
  using e_2f_t      = ripple::StorageElement<float, 2>;
  using e_b_t       = bool;
  using storage_t   = ripple::StridedStorageView<e_3i_t, e_2f_t, e_b_t>;
  using allocator_t = typename storage_t::allocator_t;
  using space_t     = ripple::DynamicMultidimSpace<3>;

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
      }
    }
  } 

  for (const auto k : ripple::range(elements_z)) {
    for (const auto j : ripple::range(elements_y)) {
      for (const auto i : ripple::range(elements_x)) {
        auto a = allocator_t::offset(storage, space, i, j, k);
  
        // Test copy:
        auto s = a;
   
        // e_3i 
        EXPECT_EQ((s.get<0, 0>()), i);
        EXPECT_EQ((s.get<0, 1>()), i + 1);
        EXPECT_EQ((s.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((s.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(s.get<2>(), (i % 2 == 0) ? true : false);

        // Test move:
        auto s1 = std::move(s);

        // e_3i 
        EXPECT_EQ((s1.get<0, 0>()), i);
        EXPECT_EQ((s1.get<0, 1>()), i + 1);
        EXPECT_EQ((s1.get<0, 2>()), i + 2);
        // e_3f
        EXPECT_EQ((s1.get<1, 0>()), static_cast<float>(i));
        EXPECT_EQ((s1.get<1, 1>()), static_cast<float>(i + 1));
        // e_b
        EXPECT_EQ(s1.get<2>(), (i % 2 == 0) ? true : false);
      }
    }
  } 
  
  free(data);
}

#endif // RIPPLE_TESTS_STRIDED_STORAGE_TESTS_HPP
