//==--- ripple/tests/iterator/blck_iterator_tests.hpp ------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_iterator_tests.hpp
/// \brief This file defines tests for block iterator functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_ITERATOR_BLOCK_ITERATOR_TESTS_HPP
#define RIPPLE_TESTS_ITERATOR_BLOCK_ITERATOR_TESTS_HPP

#include <ripple/core/container/host_block.hpp>
#include <ripple/core/container/vec.hpp>
#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/stridable_layout.hpp>
#include <gtest/gtest.h>

// This is a test class for creating a class which can be stored in a strided
// manner in the block. This is the typical structure of any class which
// represents data to be processed.
template <typename T, typename Layout = ripple::strided_view_t>
struct BlockIterTest : ripple::StridableLayout<BlockIterTest<T, Layout>> {
  // A descriptor needs to be defined for how to store the data, and the
  // layout of the data.
  using descriptor_t = ripple::StorageDescriptor<
    Layout, ripple::StorageElement<T, 3>, int
  >;
 private:
  template <typename FriendT, typename FriendLayout>
  friend struct BlockIterTest;

  // Get the storage type from the descriptor.
  using storage_t = typename descriptor_t::storage_t;

  storage_t _storage; //!< Actual storage for the type.

 public:
  // Constructor from storage is required:
  BlockIterTest(storage_t s) : _storage(s) {}

  // Constructor from other storage types.
  template <typename FriendLayout>
  BlockIterTest(const BlockIterTest<T, FriendLayout>& other)
  : _storage{other._storage} {}

  auto storage() const {
    return _storage;
  }

  //==--- [interface] ------------------------------------------------------==//
  
  auto flag() -> int& {
    return _storage.template get<1>();
  }

  auto flag() const -> const int& {
    return _storage.template get<1>();
  }

  auto v(std::size_t i) -> T& {
    return _storage.template get<0>(i);
  }

  auto v(std::size_t i) const -> const T& {
    return _storage.template get<0>(i);
  }
};

using iter_strided_t = BlockIterTest<float>;

//==--- [dimensions] -------------------------------------------------------==//

TEST(iterator_block_iterator, can_get_dimensions_at_compile_time) {
  using space_t = ripple::DynamicMultidimSpace<3>;
  using iter_t  = ripple::BlockIterator<float, space_t>;

  float f = 1.0f;
  space_t space(10, 11, 12);
  iter_t  iter(&f, space);
  constexpr auto dims = iter.dimensions();
  
  EXPECT_EQ(dims, static_cast<decltype(dims)>(3));
}

//==--- [size] -------------------------------------------------------------==//

TEST(iterator_block_iterator, can_get_correct_sizes) {
  using space_t = ripple::DynamicMultidimSpace<3>;
  using iter_t  = ripple::BlockIterator<float, space_t>;

  float f = 1.0f;
  space_t space(10, 11, 12);
  iter_t  iter(&f, space);

  EXPECT_EQ(iter.size()             , std::size_t{10 * 11 * 12});
  EXPECT_EQ(iter.size(ripple::dim_x), std::size_t{10});
  EXPECT_EQ(iter.size(ripple::dim_y), std::size_t{11});
  EXPECT_EQ(iter.size(ripple::dim_z), std::size_t{12});
}

//==--- [copy] -------------------------------------------------------------==//

TEST(iterator_block_iterator, can_make_copies_non_stridable_types) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_block_3d_t<vec_t> b(20, 20, 10);

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto v = b(i, j, k);
        for (auto element : ripple::range(v->size())) {
          (*v)[element] = static_cast<float>(element) * j;
          EXPECT_EQ((*v)[element] , static_cast<float>(element) * j);
        }
      }
    }
  }

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto v = *b(i, j, k);
        for (auto element : ripple::range(v.size())) {
          EXPECT_EQ(v[element] , static_cast<float>(element) * j);
        }
      }
    }
  }
}

TEST(iterator_block_iterator, can_make_copies_stridable_types) {
  ripple::host_block_3d_t<iter_strided_t> b(20, 30, 15);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = -1;
        bi->v(0)   = 10.0f;
        bi->v(1)   = 20.0f;
        bi->v(2)   = 30.0f;
      }
    }
  }
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k).unwrap();

        EXPECT_EQ(bi.flag(), -1   );
        EXPECT_EQ(bi.v(0)  , 10.0f);
        EXPECT_EQ(bi.v(1)  , 20.0f);
        EXPECT_EQ(bi.v(2)  , 30.0f);
      }
    }
  }
}

using iter_contig_t = BlockIterTest<int, ripple::contiguous_view_t>;

TEST(iterator_block_iterator, can_make_copies_contig_types) {
  ripple::host_block_3d_t<iter_contig_t> b(20, 30, 15);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = -1;
        bi->v(0)   = 10;
        bi->v(1)   = 20;
        bi->v(2)   = 30;
      }
    }
  }
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k).unwrap();

        EXPECT_EQ(bi.flag(), -1);
        EXPECT_EQ(bi.v(0)  , 10);
        EXPECT_EQ(bi.v(1)  , 20);
        EXPECT_EQ(bi.v(2)  , 30);
      }
    }
  }
}

//==--- [offsetting] -------------------------------------------------------==//

TEST(iterator_block_iterator, can_offset_iterator_strided) {
  ripple::host_block_3d_t<iter_strided_t> b(20, 15, 11);

  // Set the block data:
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = -1;
        bi->v(0)   = 10.0f;
        bi->v(1)   = 20.0f;
        bi->v(2)   = 30.0f;
      }
    }
  }

  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi       = b(i, j, k);
        const auto iter_off = iter
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k);

        EXPECT_EQ(bi->flag(), -1   );
        EXPECT_EQ(bi->v(0)  , 10.0f);
        EXPECT_EQ(bi->v(1)  , 20.0f);
        EXPECT_EQ(bi->v(2)  , 30.0f);


        EXPECT_EQ(iter_off->flag(), -1   );
        EXPECT_EQ(iter_off->v(0)  , 10.0f);
        EXPECT_EQ(iter_off->v(1)  , 20.0f);
        EXPECT_EQ(iter_off->v(2)  , 30.0f);

        EXPECT_EQ((*iter_off).flag(), -1   );
        EXPECT_EQ((*iter_off).v(0)  , 10.0f);
        EXPECT_EQ((*iter_off).v(1)  , 20.0f);
        EXPECT_EQ((*iter_off).v(2)  , 30.0f);
      }
    }
  }
}

TEST(iterator_block_iterator, can_offst_iterator_contig) {
  ripple::host_block_3d_t<iter_contig_t> b(20, 30, 15);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = -1;
        bi->v(0)   = 10;
        bi->v(1)   = 20;
        bi->v(2)   = 30;
      }
    }
  }
  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi       = b(i, j, k);
        const auto iter_off = iter
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k);

        EXPECT_EQ(bi->flag(), -1);
        EXPECT_EQ(bi->v(0)  , 10);
        EXPECT_EQ(bi->v(1)  , 20);
        EXPECT_EQ(bi->v(2)  , 30);

        EXPECT_EQ((*iter_off).flag(), -1);
        EXPECT_EQ((*iter_off).v(0)  , 10);
        EXPECT_EQ((*iter_off).v(1)  , 20);
        EXPECT_EQ((*iter_off).v(2)  , 30);
      }
    }
  }
}

struct NormalTest {
  auto get() -> float& {
    return f;
  }
  auto get() const -> float {
    return f;
  }
  float f = 0.0f;
};

TEST(iterator_block_iterator, can_offset_iterator_normal) {
  ripple::host_block_3d_t<NormalTest> b(20, 30, 15);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        b(i, j, k)->get() = 22.5f;
      }
    }
  }
  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi       = b(i, j, k);
        const auto iter_off = iter
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k);

        EXPECT_EQ(bi->get()        , 22.5f);
        EXPECT_EQ(iter_off->get()  , 22.5f);
        EXPECT_EQ((*iter_off).get(), 22.5f);
      }
    }
  }
}

TEST(iterator_block_iterator, can_offset_iterator_primitive) {
  ripple::host_block_3d_t<double> b(20, 30, 15);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        *b(i, j, k) = 31.0;
      }
    }
  }
  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi       = b(i, j, k);
        const auto iter_off = iter
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k);

        EXPECT_EQ(*bi      , 31.0);
        EXPECT_EQ(*iter_off, 31.0);
      }
    }
  }
}

//===--- [shifting] --------------------------------------------------------==//


TEST(iterator_block_iterator, can_shift_iterator_strided) {
  ripple::host_block_3d_t<iter_strided_t> b(21, 32, 17);

  // Set the block data:
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = -1;
        bi->v(0)   = 10.0f;
        bi->v(1)   = 20.0f;
        bi->v(2)   = 30.0f;
      }
    }
  }

  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), -1   );
        EXPECT_EQ(bi->v(0)  , 10.0f);
        EXPECT_EQ(bi->v(1)  , 20.0f);
        EXPECT_EQ(bi->v(2)  , 30.0f);

        EXPECT_EQ(iter->flag(), -1   );
        EXPECT_EQ(iter->v(0)  , 10.0f);
        EXPECT_EQ(iter->v(1)  , 20.0f);
        EXPECT_EQ(iter->v(2)  , 30.0f);

        EXPECT_EQ((*iter).flag(), -1   );
        EXPECT_EQ((*iter).v(0)  , 10.0f);
        EXPECT_EQ((*iter).v(1)  , 20.0f);
        EXPECT_EQ((*iter).v(2)  , 30.0f);

        iter.shift(ripple::dim_x, 1);
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
    iter.shift(ripple::dim_y, -1 * b.size(ripple::dim_y));
    iter.shift(ripple::dim_z, 1);
  }
}

TEST(iterator_block_iterator, can_shift_iterator_contig) {
  ripple::host_block_3d_t<iter_contig_t> b(32, 41, 7);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = -1;
        bi->v(0)   = 10;
        bi->v(1)   = 20;
        bi->v(2)   = 30;
      }
    }
  }
  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), -1);
        EXPECT_EQ(bi->v(0)  , 10);
        EXPECT_EQ(bi->v(1)  , 20);
        EXPECT_EQ(bi->v(2)  , 30);

        EXPECT_EQ(iter->flag(), -1);
        EXPECT_EQ(iter->v(0)  , 10);
        EXPECT_EQ(iter->v(1)  , 20);
        EXPECT_EQ(iter->v(2)  , 30);

        EXPECT_EQ((*iter).flag(), -1);
        EXPECT_EQ((*iter).v(0)  , 10);
        EXPECT_EQ((*iter).v(1)  , 20);
        EXPECT_EQ((*iter).v(2)  , 30);
        iter.shift(ripple::dim_x, 1);
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
    iter.shift(ripple::dim_y, -1 * b.size(ripple::dim_y));
    iter.shift(ripple::dim_z, 1);
  }
}

TEST(iterator_block_iterator, can_shift_iterator_normal) {
  ripple::host_block_3d_t<NormalTest> b(20, 30, 15);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        b(i, j, k)->get() = 22.5f;
      }
    }
  }
  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->get()    , 22.5f);
        EXPECT_EQ(iter->get()  , 22.5f);
        EXPECT_EQ((*iter).get(), 22.5f);
        iter.shift(ripple::dim_x, 1);
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
    iter.shift(ripple::dim_y, -1 * b.size(ripple::dim_y));
    iter.shift(ripple::dim_z, 1);
  }
}

TEST(iterator_block_iterator, can_shift_iterator_primitive) {
  ripple::host_block_3d_t<double> b(19, 27, 36);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        *b(i, j, k) = 31.0;
      }
    }
  }
  auto iter = b(0, 0, 0);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(*bi  , 31.0);
        EXPECT_EQ(*iter, 31.0);
        iter.shift(ripple::dim_x, 1);
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
    iter.shift(ripple::dim_y, -1 * b.size(ripple::dim_y));
    iter.shift(ripple::dim_z, 1);
  }
}

//==--- [differencing] -----------------------------------------------------==//

TEST(iterator_block_tests, can_compute_differences_correctly_non_udt) {
  using type_t = double;
  ripple::host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = 2;
  *b(0, 1, 1) = 1; *b(2, 1, 1) = 3;  // x dim
  *b(1, 0, 1) = 1; *b(1, 2, 1) = 3;  // y dim
  *b(1, 1, 0) = 1; *b(1, 1, 2) = 3;  // z dim

  // Get the iterator to the center of the block, then compute all the
  // differences:
  auto it = b(1, 1, 1);

  // x dimension:
  EXPECT_EQ(it.backward_diff(ripple::dim_x), type_t{1});
  EXPECT_EQ(it.forward_diff(ripple::dim_x) , type_t{1});
  EXPECT_EQ(it.central_diff(ripple::dim_x) , type_t{2});

  // y dimension:
  EXPECT_EQ(it.backward_diff(ripple::dim_y), type_t{1});
  EXPECT_EQ(it.forward_diff(ripple::dim_y) , type_t{1});
  EXPECT_EQ(it.central_diff(ripple::dim_y) , type_t{2});

  // z dimension:
  EXPECT_EQ(it.backward_diff(ripple::dim_z), type_t{1});
  EXPECT_EQ(it.forward_diff(ripple::dim_z) , type_t{1});
  EXPECT_EQ(it.central_diff(ripple::dim_z) , type_t{2});
}

// This tests that the iterator differencing works correctly when the type
// is an array type, and when the data is strided  non-owned.
TEST(iterator_block_tests, can_compute_differences_correctly_contig_owned) {
  using namespace ripple;
  using type_t  = Vector<int, 3, contiguous_owned_t>;
  using owned_t = Vector<int, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(2, 2, 2);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(3, 3, 3); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(3, 3, 3); // dim z

  unrolled_for<3>([&] (auto d) {
    auto diff = b(1, 1, 1).backward_diff(d);
    EXPECT_EQ(diff[0], 1); EXPECT_EQ(diff[1], 1); EXPECT_EQ(diff[2], 1);

    diff = b(1, 1, 1).forward_diff(d);
    EXPECT_EQ(diff[0], 1); EXPECT_EQ(diff[1], 1); EXPECT_EQ(diff[2], 1);

    diff = b(1, 1, 1).central_diff(d);
    EXPECT_EQ(diff[0], 2); EXPECT_EQ(diff[1], 2); EXPECT_EQ(diff[2], 2);

  });
}

// This tests that the iterator differencing works correctly when the type
// is an array type, and when the data is contiguously non-owned.
TEST(iterator_block_tests, can_compute_differences_correctly_contig_view) {
  using namespace ripple;
  using type_t  = Vector<int, 3, contiguous_view_t>;
  using owned_t = Vector<int, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(2, 2, 2);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(3, 3, 3); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(3, 3, 3); // dim z

  unrolled_for<3>([&] (auto d) {
    auto diff = b(1, 1, 1).backward_diff(d);
    EXPECT_EQ(diff[0], 1); EXPECT_EQ(diff[1], 1); EXPECT_EQ(diff[2], 1);

    diff = b(1, 1, 1).forward_diff(d);
    EXPECT_EQ(diff[0], 1); EXPECT_EQ(diff[1], 1); EXPECT_EQ(diff[2], 1);

    diff = b(1, 1, 1).central_diff(d);
    EXPECT_EQ(diff[0], 2); EXPECT_EQ(diff[1], 2); EXPECT_EQ(diff[2], 2);

  });
}

// This tests that the iterator differencing works correctly when the type
// is an array type, and when the data is strided  non-owned.
TEST(iterator_block_tests, can_compute_differences_correctly_strided_view) {
  using namespace ripple;
  using type_t  = Vector<int, 3, strided_view_t>;
  using owned_t = Vector<int, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(2, 2, 2);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(3, 3, 3); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(3, 3, 3); // dim z

  unrolled_for<3>([&] (auto d) {
    auto diff = b(1, 1, 1).backward_diff(d);
    EXPECT_EQ(diff[0], 1); EXPECT_EQ(diff[1], 1); EXPECT_EQ(diff[2], 1);

    diff = b(1, 1, 1).forward_diff(d);
    EXPECT_EQ(diff[0], 1); EXPECT_EQ(diff[1], 1); EXPECT_EQ(diff[2], 1);

    diff = b(1, 1, 1).central_diff(d);
    EXPECT_EQ(diff[0], 2); EXPECT_EQ(diff[1], 2); EXPECT_EQ(diff[2], 2);

  });
}

//==--- [gradient] ---------------------------------------------------------==//

TEST(iterator_block_tests, can_compute_gradients_correctly_non_udt) {
  using type_t = double;
  ripple::host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = 2;
  *b(0, 1, 1) = 1; *b(2, 1, 1) = 3;  // x dim
  *b(1, 0, 1) = 1; *b(1, 2, 1) = 3;  // y dim
  *b(1, 1, 0) = 1; *b(1, 1, 2) = 3;  // z dim

  // Get the iterator to the center of the block, then compute all the
  // differences:
  auto it = b(1, 1, 1);

  // x dimension:
  auto dh   = type_t{1};
  auto grad = it.grad(dh);
  EXPECT_EQ(grad[ripple::dim_x], type_t{1});
  EXPECT_EQ(grad[ripple::dim_y], type_t{1});
  EXPECT_EQ(grad[ripple::dim_z], type_t{1});
}

// This tests that the iterator gradietn works correctly when the type
// is an array type, and when the data is strided non-owned.
TEST(iterator_block_tests, can_compute_gradients_correctly_contig_owned) {
  using namespace ripple;
  using type_t  = Vector<float, 3, contiguous_owned_t>;
  using owned_t = Vector<float, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(2, 2, 2);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(5, 5, 5); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(5, 5, 5); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(5, 5, 5); // dim z

  const auto dh = double{1};
  auto grad = b(1, 1, 1).grad(dh);

  // Gradient is a vector of vectors, in each dimension:
  unrolled_for<3>([&] (auto d) {
    constexpr auto dim = size_t{d};
    EXPECT_EQ(grad[dim][0], 2); 
    EXPECT_EQ(grad[dim][1], 2);
    EXPECT_EQ(grad[dim][2], 2);
  });
}

// This tests that the iterator gradietn works correctly when the type
// is an array type, and when the data is strided non-owned.
TEST(iterator_block_tests, can_compute_gradients_correctly_contig_view) {
  using namespace ripple;
  using data_t  = float;
  using type_t  = Vector<data_t, 3, contiguous_view_t>;
  using owned_t = Vector<data_t, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(2, 2, 2);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(3, 3, 3); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(3, 3, 3); // dim z

  const auto dh = double{1};
  auto grad = b(1, 1, 1).grad(dh);

  // Gradient is a vector of vectors, in each dimension:
  unrolled_for<3>([&] (auto d) {
    constexpr auto dim = size_t{d};
    EXPECT_EQ(grad[dim][0], 1); 
    EXPECT_EQ(grad[dim][1], 1);
    EXPECT_EQ(grad[dim][2], 1);
  });
}

// This tests that the iterator gradietn works correctly when the type
// is an array type, and when the data is strided non-owned.
TEST(iterator_block_tests, can_compute_gradients_correctly_strided_view) {
  using namespace ripple;
  using data_t  = int;
  using type_t  = Vector<data_t, 3, strided_view_t>;
  using owned_t = Vector<data_t, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(2, 2, 2);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(3, 3, 3); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(3, 3, 3); // dim z

  const auto dh = data_t{1};
  auto grad = b(1, 1, 1).grad(dh);

  // Gradient is a vector of vectors, in each dimension:
  unrolled_for<3>([&] (auto d) {
    constexpr auto dim = size_t{d};
    EXPECT_EQ(grad[dim][0], 1); 
    EXPECT_EQ(grad[dim][1], 1);
    EXPECT_EQ(grad[dim][2], 1);
  });
}

//==--- [normal] -----------------------------------------------------------==//

TEST(iterator_block_tests, can_compute_normal_correctly_non_udt) {
  using type_t = double;
  ripple::host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = 4;
  *b(0, 1, 1) = 1; *b(2, 1, 1) = 7;  // x dim
  *b(1, 0, 1) = 1; *b(1, 2, 1) = 3;  // y dim
  *b(1, 1, 0) = 1; *b(1, 1, 2) = 5;  // z dim

  // Get the iterator to the center of the block, then compute all the
  // differences:
  auto it = b(1, 1, 1);

  // x dimension:
  auto dh        = type_t{1};
  auto norm      = it.norm(dh);
  const auto mag = ripple::math::sqrt(type_t{9 + 1 + 4});
  EXPECT_EQ(norm[ripple::dim_x], -type_t{3} / mag);
  EXPECT_EQ(norm[ripple::dim_y], -type_t{1} / mag);
  EXPECT_EQ(norm[ripple::dim_z], -type_t{2} / mag);

  // Change data to be signed distance with dh = 2.
  *b(0, 1, 1) = 6; *b(2, 1, 1) = 2;  // x dim
  *b(1, 0, 1) = 6; *b(1, 2, 1) = 2;  // y dim
  *b(1, 1, 0) = 6; *b(1, 1, 2) = 2;  // z dim

  auto dh2 = type_t{2};
  norm = it.norm_sd(dh2);
  EXPECT_EQ(norm[ripple::dim_x], type_t{1});
  EXPECT_EQ(norm[ripple::dim_y], type_t{1});
  EXPECT_EQ(norm[ripple::dim_z], type_t{1});
}

// This tests that the iterator normal works correctly when the type
// is an array type, and when the data is contiguous and owned.
TEST(iterator_block_tests, can_compute_normal_correctly_contig_owned) {
  using namespace ripple;
  using data_t  = float;
  using type_t  = Vector<data_t, 3, contiguous_owned_t>;
  using owned_t = Vector<data_t, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(4, 4, 4);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(7, 7, 7); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(5, 5, 5); // dim z

  const auto dh   = data_t{1};
  auto       norm = b(1, 1, 1).norm(dh);
  const auto mag  = ripple::math::sqrt(data_t{9 + 1 + 4});

  // norm is a vector of vectors.
  unrolled_for<3>([&] (auto e) {
    constexpr auto elem = size_t{e};
    EXPECT_EQ(norm[ripple::dim_x][elem], -data_t{3} / mag); 
    EXPECT_EQ(norm[ripple::dim_y][elem], -data_t{1} / mag);
    EXPECT_EQ(norm[ripple::dim_z][elem], -data_t{2} / mag);
  });
}

// This tests that the iterator normal works correctly when the type
// is an array type, and when the data is contiguous and non-owned.
TEST(iterator_block_tests, can_compute_normal_correctly_contig_view) {
  using namespace ripple;
  using data_t  = float;
  using type_t  = Vector<data_t, 3, contiguous_view_t>;
  using owned_t = Vector<data_t, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(4, 4, 4);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(7, 7, 7); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(5, 5, 5); // dim z

  const auto dh   = data_t{1};
  auto       norm = b(1, 1, 1).norm(dh);
  const auto mag  = ripple::math::sqrt(data_t{9 + 1 + 4});

  // norm is a vector of vectors.
  unrolled_for<3>([&] (auto e) {
    constexpr auto elem = size_t{e};
    EXPECT_EQ(norm[ripple::dim_x][elem], -data_t{3} / mag); 
    EXPECT_EQ(norm[ripple::dim_y][elem], -data_t{1} / mag);
    EXPECT_EQ(norm[ripple::dim_z][elem], -data_t{2} / mag);
  });
}

// This tests that the iterator normal works correctly when the type
// is an array type, and when the data is strided and non-owned.
TEST(iterator_block_tests, can_compute_normal_correctly_strided_view) {
  using namespace ripple;
  using data_t  = float;
  using type_t  = Vector<data_t, 3, strided_view_t>;
  using owned_t = Vector<data_t, 3, contiguous_owned_t>;
  host_block_3d_t<type_t> b(3, 3, 3);
  
  // Set the data:
  *b(1, 1, 1) = owned_t(4, 4, 4);
  *b(0, 1, 1) = owned_t(1, 1, 1); *b(2, 1, 1) = owned_t(7, 7, 7); // dim x
  *b(1, 0, 1) = owned_t(1, 1, 1); *b(1, 2, 1) = owned_t(3, 3, 3); // dim y
  *b(1, 1, 0) = owned_t(1, 1, 1); *b(1, 1, 2) = owned_t(5, 5, 5); // dim z

  const auto dh   = data_t{1};
  auto       norm = b(1, 1, 1).norm(dh);
  const auto mag  = ripple::math::sqrt(data_t{9 + 1 + 4});

  // norm is a vector of vectors.
  unrolled_for<3>([&] (auto e) {
    constexpr auto elem = size_t{e};
    EXPECT_EQ(norm[ripple::dim_x][elem], -data_t{3} / mag); 
    EXPECT_EQ(norm[ripple::dim_y][elem], -data_t{1} / mag);
    EXPECT_EQ(norm[ripple::dim_z][elem], -data_t{2} / mag);
  });
}

#endif // RIPPLE_TESTS_ITERATOR_BLOCK_ITERATOR_HPP
