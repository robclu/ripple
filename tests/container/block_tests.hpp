//==--- ripple/tests/container/block_tests.hpp ------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_tests.hpp
/// \brief This file defines tests for block functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_BLOCK_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_BLOCK_TESTS_HPP

#include <ripple/container/block_traits.hpp>
#include <ripple/container/host_block.hpp>
#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/stridable_layout.hpp>
#include <ripple/utility/dim.hpp>
#include <gtest/gtest.h>

//==--- [creation] ---------------------------------------------------------==//

TEST(container_block, can_create_block_1d) {
  ripple::host_block_1d_t<float> b(20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20));
}

TEST(container_block, can_create_block_2d) {
  ripple::host_block_2d_t<int> b(20, 20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20 * 20));
}

TEST(container_block, can_create_block_3d) {
  ripple::host_block_3d_t<double> b(20, 20, 20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20 * 20 * 20));
}

//==-- [access simple types] -----------------------------------------------==//

TEST(container_block, can_access_simple_elements_1d) {
  ripple::host_block_1d_t<float> b(20);
  for (auto i : ripple::range(b.size())) {
    *b(i) = static_cast<float>(i);
  }
  for (auto i : ripple::range(b.size())) {
    EXPECT_EQ(*b(i), static_cast<float>(i));
  }
}

TEST(container_block, can_access_simple_elements_2d) {
  ripple::host_block_2d_t<float> b(20, 20);
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      *b(i, j) = static_cast<float>(i);
    }
  }
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      EXPECT_EQ(*b(i, j), static_cast<float>(i));
    }
  }
}

TEST(container_block, can_access_simple_elements_3d) {
  ripple::host_block_3d_t<float> b(20, 20, 20);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        *b(i, j, k) = static_cast<float>(i);
      }
    }
  }
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        EXPECT_EQ(*b(i, j, k), static_cast<float>(i));
      }
    }
  }
}

//==--- [access stridable layout types] ------------------------------------==//

// This is a test class for creating a class which can be stored in a strided
// manner in the block. This is the typical structure of any class which
// represents data to be processed.
template <typename T, typename Layout = ripple::strided_view_t>
struct BlockTest : ripple::StridableLayout<BlockTest<T, Layout>> {
  // A descriptor needs to be defined for how to store the data, and the
  // layout of the data.
  using descriptor_t = ripple::StorageDescriptor<
    Layout, ripple::StorageElement<T, 3>, int
  >;
 private:
  // Get the storage type from the descriptor.
  using storage_t = typename descriptor_t::storage_t;
  storage_t _storage;

 public:
  // Constructor from storage is required:
  BlockTest(storage_t s) : _storage(s) {}

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

using test_t = BlockTest<float>;

TEST(container_block, can_access_stridable_layout_elements_1d) {
  ripple::host_block_1d_t<test_t> b(20);
  for (auto i : ripple::range(b.size())) {
    auto bi = b(i);

    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  }
  for (auto i : ripple::range(b.size())) {
    const auto bi = b(i);

    EXPECT_EQ(bi->flag(), -1   );
    EXPECT_EQ(bi->v(0)  , 10.0f);
    EXPECT_EQ(bi->v(1)  , 20.0f);
    EXPECT_EQ(bi->v(2)  , 30.0f);
  }
}

TEST(container_block, can_access_stridable_layout_elements_2d) {
  ripple::host_block_2d_t<test_t> b(20, 30);
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      auto bi = b(i, j);

      bi->flag() = -1;
      bi->v(0)   = 10.0f;
      bi->v(1)   = 20.0f;
      bi->v(2)   = 30.0f;
    }
  }
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      const auto bi = b(i, j);

      EXPECT_EQ(bi->flag(), -1   );
      EXPECT_EQ(bi->v(0)  , 10.0f);
      EXPECT_EQ(bi->v(1)  , 20.0f);
      EXPECT_EQ(bi->v(2)  , 30.0f);
    }
  }
}

TEST(container_block, can_access_stridable_layout_elements_3d) {
  ripple::host_block_3d_t<test_t> b(20, 30, 15);
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
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), -1   );
        EXPECT_EQ(bi->v(0)  , 10.0f);
        EXPECT_EQ(bi->v(1)  , 20.0f);
        EXPECT_EQ(bi->v(2)  , 30.0f);
      }
    }
  }
}

//==--- [access vec types] -------------------------------------------------==//

TEST(container_block, can_access_vec_elements_1d) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_block_1d_t<vec_t> b(3);

  for (auto i : ripple::range(b.size())) {
    auto v = b(i);
    for (auto element : ripple::range(v->size())) {
      (*v)[element] = static_cast<float>(element) * i;
      EXPECT_EQ((*v)[element] , static_cast<float>(element) * i);
    }
  }


  for (auto i : ripple::range(b.size())) {
    const auto v = b(i);
    for (auto element : ripple::range(v->size())) {
      EXPECT_EQ((*v)[element] , static_cast<float>(element) * i);
    }
  }

}

TEST(container_block, can_access_vec_elements_2d) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_block_2d_t<vec_t> b(20, 20);

  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      auto v = b(i, j);
      for (auto element : ripple::range(v->size())) {
        (*v)[element] = static_cast<float>(element) * i;
      }
    }
  }

  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      const auto v = b(i, j);
      for (auto element : ripple::range(v->size())) {
        EXPECT_EQ((*v)[element] , static_cast<float>(element) * i);
      }
    }
  }
}

TEST(container_block, can_access_vec_elements_3d) {
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
        const auto v = b(i, j, k);
        for (auto element : ripple::range(v->size())) {
          EXPECT_EQ((*v)[element] , static_cast<float>(element) * j);
        }
      }
    }
  }
}

#endif // RIPPLE_TESTS_CONTAINER_BLOCK_TESTS_HPP

