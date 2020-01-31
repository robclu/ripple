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
#include <ripple/functional/invoke.hpp>
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

//==--- [invoke] -----------------------------------------------------------==//

TEST(container_block, can_invoke_1d) {
  ripple::host_block_1d_t<test_t> b(4121);

  ripple::invoke(b, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

  for (auto i : ripple::range(b.size(ripple::dim_x))) {
    const auto bi = b(i);

    EXPECT_EQ(bi->flag(), -1   );
    EXPECT_EQ(bi->v(0)  , 10.0f);
    EXPECT_EQ(bi->v(1)  , 20.0f);
    EXPECT_EQ(bi->v(2)  , 30.0f);
  }
}

TEST(container_block, can_invoke_2d) {
  ripple::host_block_2d_t<test_t> b(412, 371);

  ripple::invoke(b, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

TEST(container_block, can_invoke_3d) {
  ripple::host_block_3d_t<test_t> b(41, 37, 22);

  ripple::invoke(b, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

TEST(container_block, can_invoke_3d_with_padding) {
  ripple::host_block_3d_t<test_t> b(2, 41, 37, 22);

  ripple::invoke(b, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

  const auto internal_size = std::size_t{41 * 37 * 22};
  EXPECT_EQ(b.size()   , internal_size);
  EXPECT_EQ(b.padding(), std::size_t{2});
}

//==--- [modify padding] ---------------------------------------------------==//

TEST(container_block, can_modify_padding) {
  ripple::host_block_3d_t<test_t> b(2, 41, 37, 22);

  ripple::invoke(b, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

  const auto internal_size = std::size_t{41 * 37 * 22};
  EXPECT_EQ(b.size()   , internal_size);
  EXPECT_EQ(b.padding(), std::size_t{2});

  b.set_padding(3);
  b.reallocate();

  ripple::invoke(b, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

  EXPECT_EQ(b.size()   , internal_size);
  EXPECT_EQ(b.padding(), std::size_t{3});
}

//==--- [invoke multiple blocks] -------------------------------------------==//

TEST(container_block, can_invoke_multiple_blocks_1d) {
  constexpr auto size_x = 4712;
  ripple::host_block_1d_t<test_t> b1(size_x);
  ripple::host_block_1d_t<test_t> b2(size_x);

  ripple::invoke(b1, b2, [] (auto bi1, auto bi2) {
    bi1->flag() = -1;
    bi1->v(0)   = 10.0f;
    bi1->v(1)   = 20.0f;
    bi1->v(2)   = 30.0f;

    bi2->flag() = bi1->flag();
    bi2->v(0)   = bi1->v(0);
    bi2->v(1)   = bi1->v(1);
    bi2->v(2)   = bi1->v(2);
  });

  for (auto i : ripple::range(b2.size(ripple::dim_x))) {
    const auto bi1 = b1(i);
    const auto bi2 = b2(i);

    EXPECT_EQ(bi1->flag(), -1   );
    EXPECT_EQ(bi1->v(0)  , 10.0f);
    EXPECT_EQ(bi1->v(1)  , 20.0f);
    EXPECT_EQ(bi1->v(2)  , 30.0f);

    EXPECT_EQ(bi2->flag(), -1   );
    EXPECT_EQ(bi2->v(0)  , 10.0f);
    EXPECT_EQ(bi2->v(1)  , 20.0f);
    EXPECT_EQ(bi2->v(2)  , 30.0f);
  }
}

TEST(container_block, can_invoke_multiple_blocks_2d) {
  constexpr auto size_x = 412;
  constexpr auto size_y = 571;
  ripple::host_block_2d_t<test_t> b1(size_x, size_y);
  ripple::host_block_2d_t<test_t> b2(size_x, size_y);

  ripple::invoke(b1, b2, [] (auto bi1, auto bi2) {
    bi1->flag() = -1;
    bi1->v(0)   = 10.0f;
    bi1->v(1)   = 20.0f;
    bi1->v(2)   = 30.0f;

    bi2->flag() = bi1->flag();
    bi2->v(0)   = bi1->v(0);
    bi2->v(1)   = bi1->v(1);
    bi2->v(2)   = bi1->v(2);
  });

  for (auto j : ripple::range(b2.size(ripple::dim_y))) {
    for (auto i : ripple::range(b2.size(ripple::dim_x))) {
      const auto bi1 = b1(i, j);
      const auto bi2 = b2(i, j);

      EXPECT_EQ(bi1->flag(), -1   );
      EXPECT_EQ(bi1->v(0)  , 10.0f);
      EXPECT_EQ(bi1->v(1)  , 20.0f);
      EXPECT_EQ(bi1->v(2)  , 30.0f);

      EXPECT_EQ(bi2->flag(), -1   );
      EXPECT_EQ(bi2->v(0)  , 10.0f);
      EXPECT_EQ(bi2->v(1)  , 20.0f);
      EXPECT_EQ(bi2->v(2)  , 30.0f);
    }
  }
}


TEST(container_block, can_invoke_multiple_blocks_3d) {
  constexpr auto size_x = 61;
  constexpr auto size_y = 57;
  constexpr auto size_z = 12;
  ripple::host_block_3d_t<test_t> b1(size_x, size_y, size_z);
  ripple::host_block_3d_t<test_t> b2(size_x, size_y, size_z);

  ripple::invoke(b1, b2, [] (auto bi1, auto bi2) {
    bi1->flag() = -1;
    bi1->v(0)   = 10.0f;
    bi1->v(1)   = 20.0f;
    bi1->v(2)   = 30.0f;

    bi2->flag() = bi1->flag();
    bi2->v(0)   = bi1->v(0);
    bi2->v(1)   = bi1->v(1);
    bi2->v(2)   = bi1->v(2);
  });

  for (auto k : ripple::range(b2.size(ripple::dim_z))) {
    for (auto j : ripple::range(b2.size(ripple::dim_y))) {
      for (auto i : ripple::range(b2.size(ripple::dim_x))) {
        const auto bi1 = b1(i, j, k);
        const auto bi2 = b2(i, j, k);

        EXPECT_EQ(bi1->flag(), -1   );
        EXPECT_EQ(bi1->v(0)  , 10.0f);
        EXPECT_EQ(bi1->v(1)  , 20.0f);
        EXPECT_EQ(bi1->v(2)  , 30.0f);

        EXPECT_EQ(bi2->flag(), -1   );
        EXPECT_EQ(bi2->v(0)  , 10.0f);
        EXPECT_EQ(bi2->v(1)  , 20.0f);
        EXPECT_EQ(bi2->v(2)  , 30.0f);
      }
    }
  }
}

//==--- [invoke blocked] ---------------------------------------------------==//

TEST(container_block, can_invoke_with_exec_params_1d) {
  constexpr auto size_x = 141;
  ripple::host_block_1d_t<test_t> b(size_x);
  ripple::StaticExecParams<32, 1, 1> params;

  ripple::invoke(b, params, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

  for (auto i : ripple::range(b.size(ripple::dim_x))) {
    const auto bi = b(i);

    EXPECT_EQ(bi->flag(), -1   );
    EXPECT_EQ(bi->v(0)  , 10.0f);
    EXPECT_EQ(bi->v(1)  , 20.0f);
    EXPECT_EQ(bi->v(2)  , 30.0f);
  }
}

TEST(container_block, can_invoke_with_exec_params_2d) {
  constexpr auto size_x = 151;
  constexpr auto size_y = 37;
  ripple::host_block_2d_t<test_t> b(size_x, size_y);
  ripple::StaticExecParams<16, 16, 1> params;


  ripple::invoke(b, params, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

TEST(container_block, can_invoke_with_exec_params_3d) {
  constexpr auto size_x = 31;
  constexpr auto size_y = 41;
  constexpr auto size_z = 27;
  ripple::host_block_3d_t<test_t> b(size_x, size_y, size_z);
  ripple::StaticExecParams<8, 8, 8> params;

  ripple::invoke(b, params, [] (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

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

//==--- [access begin] -----------------------------------------------------==//

TEST(container_block, can_access_block_beginning) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_block_3d_t<vec_t> b(20, 20, 10);

  (*b(0, 0, 0))[0] = 22.5f;
  (*b(0, 0, 0))[1] = 32.5f;
  (*b(0, 0, 0))[2] = 42.5f;

  auto start = b.begin();
  EXPECT_EQ((*start)[0], (*b(0, 0, 0))[0]);
  EXPECT_EQ((*start)[1], (*b(0, 0, 0))[1]);
  EXPECT_EQ((*start)[2], (*b(0, 0, 0))[2]);
}

//==--- [copying] ----------------------------------------------------------==//

TEST(container_block, can_copy_strided) {
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

  auto b1(b);
  auto b2 = b;
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi  = b(i, j, k);
        const auto bi1 = b1(i, j, k);
        const auto bi2 = b2(i, j, k);

        EXPECT_EQ(bi->flag(), -1   );
        EXPECT_EQ(bi->v(0)  , 10.0f);
        EXPECT_EQ(bi->v(1)  , 20.0f);
        EXPECT_EQ(bi->v(2)  , 30.0f);

        EXPECT_EQ(bi1->flag(), -1   );
        EXPECT_EQ(bi1->v(0)  , 10.0f);
        EXPECT_EQ(bi1->v(1)  , 20.0f);
        EXPECT_EQ(bi1->v(2)  , 30.0f);

        EXPECT_EQ(bi2->flag(), -1   );
        EXPECT_EQ(bi2->v(0)  , 10.0f);
        EXPECT_EQ(bi2->v(1)  , 20.0f);
        EXPECT_EQ(bi2->v(2)  , 30.0f);
      }
    }
  }
}

TEST(container_block, can_copy_vec) {
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

  auto b1(b);
  auto b2 = b;
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto v1 = b1(i, j, k);
        const auto v2 = b2(i, j, k);
        for (auto element : ripple::range(v1->size())) {
          EXPECT_EQ((*v1)[element] , static_cast<float>(element) * j);
          EXPECT_EQ((*v2)[element] , static_cast<float>(element) * j);
        }
      }
    }
  }
}

TEST(container_block, can_copy_simple) {
  ripple::host_block_3d_t<float> b(20, 20, 20);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        *b(i, j, k) = static_cast<float>(i);
      }
    }
  }
  auto b1(b);
  auto b2 = b;
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        EXPECT_EQ(*b1(i, j, k), static_cast<float>(i));
        EXPECT_EQ(*b2(i, j, k), static_cast<float>(i));
      }
    }
  }
}

//==--- [moving] -----------------------------------------------------------==//

TEST(container_block, can_move_strided) {
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

  auto b1(std::move(b));
  for (auto k : ripple::range(b1.size(ripple::dim_z))) {
    for (auto j : ripple::range(b1.size(ripple::dim_y))) {
      for (auto i : ripple::range(b1.size(ripple::dim_x))) {
        const auto bi  = b1(i, j, k);

        EXPECT_EQ(bi->flag(), -1   );
        EXPECT_EQ(bi->v(0)  , 10.0f);
        EXPECT_EQ(bi->v(1)  , 20.0f);
        EXPECT_EQ(bi->v(2)  , 30.0f);
      }
    }
  }
  auto b2 = std::move(b1);
  for (auto k : ripple::range(b2.size(ripple::dim_z))) {
    for (auto j : ripple::range(b2.size(ripple::dim_y))) {
      for (auto i : ripple::range(b2.size(ripple::dim_x))) {
        const auto bi  = b2(i, j, k);

        EXPECT_EQ(bi->flag(), -1   );
        EXPECT_EQ(bi->v(0)  , 10.0f);
        EXPECT_EQ(bi->v(1)  , 20.0f);
        EXPECT_EQ(bi->v(2)  , 30.0f);
      }
    }
  }
}

TEST(container_block, can_move_vec) {
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

  auto b1(std::move(b));
  for (auto k : ripple::range(b1.size(ripple::dim_z))) {
    for (auto j : ripple::range(b1.size(ripple::dim_y))) {
      for (auto i : ripple::range(b1.size(ripple::dim_x))) {
        const auto v1 = b1(i, j, k);
        for (auto element : ripple::range(v1->size())) {
          EXPECT_EQ((*v1)[element] , static_cast<float>(element) * j);
        }
      }
    }
  }
  auto b2 = std::move(b1);
  for (auto k : ripple::range(b2.size(ripple::dim_z))) {
    for (auto j : ripple::range(b2.size(ripple::dim_y))) {
      for (auto i : ripple::range(b2.size(ripple::dim_x))) {
        const auto v1 = b2(i, j, k);
        for (auto element : ripple::range(v1->size())) {
          EXPECT_EQ((*v1)[element] , static_cast<float>(element) * j);
        }
      }
    }
  }
}

TEST(container_block, can_move_simple) {
  ripple::host_block_3d_t<float> b(20, 20, 20);
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        *b(i, j, k) = static_cast<float>(i);
      }
    }
  }
  auto b1(std::move(b));
  for (auto k : ripple::range(b1.size(ripple::dim_z))) {
    for (auto j : ripple::range(b1.size(ripple::dim_y))) {
      for (auto i : ripple::range(b1.size(ripple::dim_x))) {
        EXPECT_EQ(*b1(i, j, k), static_cast<float>(i));
      }
    }
  }
  auto b2 = std::move(b1);
  for (auto k : ripple::range(b2.size(ripple::dim_z))) {
    for (auto j : ripple::range(b2.size(ripple::dim_y))) {
      for (auto i : ripple::range(b2.size(ripple::dim_x))) {
        EXPECT_EQ(*b2(i, j, k), static_cast<float>(i));
      }
    }
  }
}

//==--- [resize] -----------------------------------------------------------==//

TEST(container_block, can_resize_block) {
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

  b.resize(15, 20, 22);

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = 17;;
        bi->v(0)   = 12.0f;
        bi->v(1)   = 21.0f;
        bi->v(2)   = 37.0f;
      }
    }
  }
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), 17   );
        EXPECT_EQ(bi->v(0)  , 12.0f);
        EXPECT_EQ(bi->v(1)  , 21.0f);
        EXPECT_EQ(bi->v(2)  , 37.0f);
      }
    }
  }

  b.resize(13, 20);
  EXPECT_EQ(b.size(ripple::dim_x), std::size_t{13});
  EXPECT_EQ(b.size(ripple::dim_y), std::size_t{20});

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = 14;;
        bi->v(0)   = 13.0f;
        bi->v(1)   = 22.0f;
        bi->v(2)   = 77.0f;
      }
    }
  }
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), 14   );
        EXPECT_EQ(bi->v(0)  , 13.0f);
        EXPECT_EQ(bi->v(1)  , 22.0f);
        EXPECT_EQ(bi->v(2)  , 77.0f);
      }
    }
  }

  b.resize_dim(ripple::dim_x, 20);
  b.resize_dim(ripple::dim_y, 120);
  b.resize_dim(ripple::dim_z, 31);
  b.reallocate();
  EXPECT_EQ(b.size(ripple::dim_x), std::size_t{20});
  EXPECT_EQ(b.size(ripple::dim_y), std::size_t{120});
  EXPECT_EQ(b.size(ripple::dim_z), std::size_t{31});

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        auto bi = b(i, j, k);

        bi->flag() = 12;;
        bi->v(0)   = 13.0f;
        bi->v(1)   = 27.0f;
        bi->v(2)   = 77.0f;
      }
    }
  }
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), 12   );
        EXPECT_EQ(bi->v(0)  , 13.0f);
        EXPECT_EQ(bi->v(1)  , 27.0f);
        EXPECT_EQ(bi->v(2)  , 77.0f);
      }
    }
  }
}

//==--- [move host to device] ----------------------------------------------==//

TEST(container_block, can_copy_host_to_device) {
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

  // Copy to device, and then back, to make sure it can go both ways:
  auto b_dev  = b.as_device();
  auto b_host = b_dev.as_host();

  for (auto k : ripple::range(b_host.size(ripple::dim_z))) {
    for (auto j : ripple::range(b_host.size(ripple::dim_y))) {
      for (auto i : ripple::range(b_host.size(ripple::dim_x))) {
        EXPECT_EQ(*b_host(i, j, k), static_cast<float>(i));
      }
    }
  }
}

#endif // RIPPLE_TESTS_CONTAINER_BLOCK_TESTS_HPP

