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

#include <ripple/container/host_block.hpp>
#include <ripple/container/vec.hpp>
#include <ripple/iterator/block_iterator.hpp>
#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/stridable_layout.hpp>
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
        }
      }
    }
  }

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto v = b(i, j, k).unwrap();
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
  ripple::host_block_3d_t<iter_strided_t> b(29, 17, 11);

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
        iter.shift(ripple::dim_x, 1);

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
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
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
        const auto bi       = b(i, j, k);
        iter.shift(ripple::dim_x, 1);

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
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
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
        const auto bi       = b(i, j, k);
        iter.shift(ripple::dim_x, 1);

        EXPECT_EQ(bi->get()        , 22.5f);
        EXPECT_EQ(iter->get()  , 22.5f);
        EXPECT_EQ((*iter).get(), 22.5f);
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
    iter.shift(ripple::dim_z, 1);
  }
}

TEST(iterator_block_iterator, can_shift_iterator_primitive) {
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
        iter.shift(ripple::dim_x, i);

        EXPECT_EQ(*bi  , 31.0);
        EXPECT_EQ(*iter, 31.0);
      }
      iter.shift(ripple::dim_x, -1 * b.size(ripple::dim_x));
      iter.shift(ripple::dim_y, 1);
    }
    iter.shift(ripple::dim_z, 1);
  }
}

#endif // RIPPLE_TESTS_ITERATOR_BLOCK_ITERATOR_HPP
