//==--- ripple/tests/container/device_block_tests.hpp ------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_block_tests.hpp
/// \brief This file defines tests for device block functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_DEVICE_BLOCK_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_DEVICE_BLOCK_TESTS_HPP

#include <ripple/container/block_traits.hpp>
#include <ripple/container/device_block.hpp>
#include <ripple/execution/static_execution_params.hpp>
#include <ripple/functional/invoke.hpp>
#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/stridable_layout.hpp>
#include <ripple/utility/dim.hpp>
#include <gtest/gtest.h>

TEST(container_device_block, can_create_block_1d) {
  ripple::device_block_1d_t<float> b(20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20));
}

TEST(container_device_block, can_create_block_2d) {
  ripple::device_block_2d_t<int> b(20, 20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20 * 20));
}

TEST(container_device_block, can_create_block_3d) {
  ripple::device_block_3d_t<double> b(20, 20, 20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20 * 20 * 20));
}

//==--- [access] -----------------------------------------------------------==//

TEST(container_device_block, can_access_simple_elements_1d) {
  ripple::host_block_1d_t<float> b_host(400);
  ripple::device_block_1d_t<float> b_dev(400);
      
  for (auto i : ripple::range(b_host.size())) {
    *b_host(i) = static_cast<float>(i) + 5.0f;
  };
  for (auto i : ripple::range(b_host.size())) {
    EXPECT_EQ(*b_host(i), static_cast<float>(i) + 5.0f);
  }
  b_dev = b_host;

  ripple::invoke(b_dev, [] ripple_host_device (auto e_it) {
    *e_it += global_idx(ripple::dim_x) + 10.0f;;
  });

  auto b = b_dev.as_host();
  for (auto i : ripple::range(b.size())) {
    EXPECT_EQ(*b(i), static_cast<float>(i) * 2 + 15.0f);
  }
}

TEST(container_device_block, can_access_simple_elements_2d) {
  ripple::device_block_2d_t<float> b_dev(92, 91);

  ripple::invoke(b_dev, [] ripple_host_device (auto e) {
    *e = global_idx(ripple::dim_x);
  });

  auto b = b_dev.as_host();
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      EXPECT_EQ(*b(i, j), static_cast<float>(i));
    }
  }
}

TEST(container_device_block, can_access_simple_elements_3d) {
  ripple::device_block_3d_t<float> b_dev(250, 250, 250);

  ripple::invoke(b_dev, [] ripple_host_device (auto e) {
    *e = ripple::global_idx(ripple::dim_x);
  });

  auto b = b_dev.as_host();
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        EXPECT_EQ(*b(i, j, k), static_cast<float>(i));
      }
    }
  }
}

//==--- [invoke with exec params] ------------------------------------------==//

TEST(container_device_block, exec_params_simple_elements_1d) {
  ripple::host_block_1d_t<float> b_host(4000);
  ripple::device_block_1d_t<float> b_dev(4000);
    
  for (auto i : ripple::range(b_host.size())) {
    *b_host(i) = static_cast<float>(i) + 5.0f;
  };
  for (auto i : ripple::range(b_host.size())) {
    EXPECT_EQ(*b_host(i), static_cast<float>(i) + 5.0f);
  }
  b_dev = b_host;

  // Create execution parameters, 2 elements per thread:
  ripple::StaticExecParams<512, 1, 1> params; 
  ripple::invoke(b_dev, params,
    [] ripple_host_device (auto e_it) {
      *e_it += global_idx(ripple::dim_x) + 10.0f;;
    }
  );

  auto b = b_dev.as_host();
  for (auto i : ripple::range(b.size())) {
    EXPECT_EQ(*b(i), static_cast<float>(i) * 2 + 15.0f);
  }
}

TEST(container_device_block, exec_params_simple_elements_2d) {
  ripple::device_block_2d_t<float> b_dev(451, 107);
  ripple::StaticExecParams<32, 16, 1> params; 

  ripple::invoke(b_dev, params, [] ripple_host_device (auto e) {
    *e = global_idx(ripple::dim_x);
  });

  auto b = b_dev.as_host();
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      EXPECT_EQ(*b(i, j), static_cast<float>(i));
    }
  }
}

TEST(container_device_block, exec_params_simple_elements_3d) {
  ripple::device_block_3d_t<float> b_dev(250, 250, 250);
  ripple::StaticExecParams<16, 8, 4> params; 

  ripple::invoke(b_dev, params, [] ripple_host_device (auto e) {
    *e = ripple::global_idx(ripple::dim_x);
  });

  auto b = b_dev.as_host();
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        EXPECT_EQ(*b(i, j, k), static_cast<float>(i));
      }
    }
  }
}

//==--- [stridable types] --------------------------------------------------==//

// This is a test class for creating a class which can be stored in a strided
// manner in the block. This is the typical structure of any class which
// represents data to be processed.
template <typename T, typename Layout = ripple::strided_view_t>
struct DeviceBlockTest : ripple::StridableLayout<DeviceBlockTest<T, Layout>> {
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
  ripple_host_device DeviceBlockTest(const storage_t& s) 
  : _storage{s}  {}


  // Assignment to copy from another block test.
  ripple_host_device auto operator=(const DeviceBlockTest& other) 
  -> DeviceBlockTest& {
    _storage.copy(other._storage);
    return *this;
  }

  //==--- [interface] ------------------------------------------------------==//

  ripple_host_device auto flag() -> int& {
    return _storage.template get<1>();
  }

  ripple_host_device auto flag() const -> const int& {
    return _storage.template get<1>();
  }

  ripple_host_device auto v(std::size_t i) -> T& {
    return _storage.template get<0>(i);
  }

  ripple_host_device auto v(std::size_t i) const -> const T& {
    return _storage.template get<0>(i);
  }
};

using dev_block_test_t = DeviceBlockTest<float>;

TEST(container_device_block, can_access_stridable_layout_elements_1d) {
  ripple::device_block_1d_t<dev_block_test_t> b_dev(200);

  ripple::invoke(b_dev, [] ripple_host_device (auto e_it) {
    e_it->flag() = -1;
    e_it->v(0)   = 4.4f;
    e_it->v(1)   = 5.5f;
    e_it->v(2)   = 6.6f;
  });

  auto b = b_dev.as_host();
  for (auto i : ripple::range(b.size())) {
    const auto bi = b(i);
    EXPECT_EQ(bi->flag(), -1  );
    EXPECT_EQ(bi->v(0)  , 4.4f);
    EXPECT_EQ(bi->v(1)  , 5.5f);
    EXPECT_EQ(bi->v(2)  , 6.6f);
  }
}

TEST(container_device_block, can_access_stridable_layout_elements_2d) {
  ripple::device_block_2d_t<dev_block_test_t> b_dev(312, 3571);
  ripple::invoke(b_dev, [] ripple_host_device (auto bi) {
    bi->flag() = -1;
    bi->v(0)   = 10.0f;
    bi->v(1)   = 20.0f;
    bi->v(2)   = 30.0f;
  });

  const auto b = b_dev.as_host();
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

TEST(container_device_block, can_access_stridable_layout_elements_3d) {
  ripple::device_block_3d_t<dev_block_test_t> b_dev(312, 171, 254);
  ripple::invoke(b_dev, [] ripple_host_device (auto bi) {
    bi->flag() = -11;
    bi->v(0)   = 11.0f;
    bi->v(1)   = 29.0f;
    bi->v(2)   = 30.4f;
  });

  const auto b = b_dev.as_host();
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ(bi->flag(), -11   );
        EXPECT_EQ(bi->v(0)  , 11.0f);
        EXPECT_EQ(bi->v(1)  , 29.0f);
        EXPECT_EQ(bi->v(2)  , 30.4f);
      }
    }
  }
}

//==--- [stridable exec params] --------------------------------------------==//

TEST(container_device_block, exec_params_stridable_elements_1d) {
  ripple::device_block_1d_t<dev_block_test_t> b_dev(200);
  ripple::StaticExecParams<512, 1, 1> params;

  ripple::invoke(b_dev, params,
    [] ripple_host_device (auto e_it) {
      const auto idx = static_cast<float>(global_idx(ripple::dim_x));
      e_it->flag() = -1;
      e_it->v(0)   = idx + 4.4f;
      e_it->v(1)   = idx + 5.5f;
      e_it->v(2)   = idx + 6.6f;
    }
  );

  auto b = b_dev.as_host();
  for (auto i : ripple::range(b.size())) {
    const auto bi = b(i);
    const auto idx = static_cast<float>(i);
    EXPECT_EQ(bi->flag(), -1  );
    EXPECT_EQ(bi->v(0)  , idx + 4.4f);
    EXPECT_EQ(bi->v(1)  , idx + 5.5f);
    EXPECT_EQ(bi->v(2)  , idx + 6.6f);
  }
}

TEST(container_device_block, exec_params_stridable_elements_2d) {
  ripple::device_block_2d_t<dev_block_test_t> b_dev(312, 3571);
  ripple::StaticExecParams<32, 32, 1> params;

  ripple::invoke(b_dev, params, [] ripple_host_device (auto bi) {
    const auto idx = static_cast<float>(global_idx(ripple::dim_x));
    bi->flag() = -1;
    bi->v(0)   = idx + 10.0f;
    bi->v(1)   = idx + 20.0f;
    bi->v(2)   = idx + 30.0f;
  });

  const auto b = b_dev.as_host();
  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      const auto bi  = b(i, j);
      const auto idx = static_cast<float>(i);
      
      EXPECT_EQ(bi->flag(), -1   );
      EXPECT_EQ(bi->v(0)  , idx + 10.0f);
      EXPECT_EQ(bi->v(1)  , idx + 20.0f);
      EXPECT_EQ(bi->v(2)  , idx + 30.0f);
    }
  }
}

TEST(container_device_block, exec_params_stridable_elements_3d) {
  ripple::device_block_3d_t<dev_block_test_t> b_dev(312, 171, 254);
  ripple::StaticExecParams<8, 8, 4> params;
  ripple::invoke(b_dev, params, [] ripple_host_device (auto bi) {
    const auto idx = static_cast<float>(global_idx(ripple::dim_x));
    bi->flag() = -11;
    bi->v(0)   = idx + 11.0f;
    bi->v(1)   = idx + 29.0f;
    bi->v(2)   = idx + 30.4f;
  });

  const auto b = b_dev.as_host();
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi  = b(i, j, k);
        const auto idx = static_cast<float>(i);

        EXPECT_EQ(bi->flag(), -11   );
        EXPECT_EQ(bi->v(0)  , idx + 11.0f);
        EXPECT_EQ(bi->v(1)  , idx + 29.0f);
        EXPECT_EQ(bi->v(2)  , idx + 30.4f);
      }
    }
  }
}

//==--- [static shared] ----------------------------------------------------==//

TEST(container_device_block, exec_params_shared_stridable_elements_3d) {
  ripple::device_block_3d_t<dev_block_test_t> b_dev(60, 12, 27);
  ripple::StaticExecParams<8, 8, 4, 0, dev_block_test_t> params;
  ripple::invoke(b_dev, params, [] ripple_host_device (auto bi, auto si) {
    const auto idx = static_cast<float>(global_idx(ripple::dim_x));

    si->flag() = -11;
    si->v(0)   = idx + 11.0f;
    si->v(1)   = idx + 29.0f;
    si->v(2)   = idx + 30.4f;

    *bi = *si;
  });

  const auto b = b_dev.as_host();
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi  = b(i, j, k);
        const auto idx = static_cast<float>(i);

        EXPECT_EQ(bi->flag(), -11   );
        EXPECT_EQ(bi->v(0)  , idx + 11.0f);
        EXPECT_EQ(bi->v(1)  , idx + 29.0f);
        EXPECT_EQ(bi->v(2)  , idx + 30.4f);
      }
    }
  }
}

TEST(container_device_block, exec_params_shared_pad_stridable_elements_3d) {
  ripple::device_block_3d_t<dev_block_test_t> b_dev(2, 2, 2);
  constexpr auto padding = 2;
  constexpr auto tol     = 1e-4;
  ripple::StaticExecParams<8, 4, 2, padding, dev_block_test_t> params;
  ripple::invoke(b_dev, params, [] ripple_host_device (auto bi, auto si) {
    const auto idx = static_cast<float>(global_idx(ripple::dim_x));
    si->flag() = -11;
    si->v(0)   = 11.0f;
    si->v(1)   = 29.0f;
    si->v(2)   = 30.4f;

    // Set the padding data:
    ripple::unrolled_for<3>([&] (auto dim) {
      if (first_thread_in_block(dim)) {
        for (auto i = std::size_t{1}; i <= si.padding(); ++i) {
          auto s = si.offset(dim, -1 * i);
          s->flag() = -11;
          s->v(0)   = 11.0f;
          s->v(1)   = 29.0f;
          s->v(2)   = 30.4f;
        }
      } else if (ripple::last_thread_in_block(dim)) {
        for (auto i = std::size_t{1}; i <= si.padding(); ++i) {
          auto s = si.offset(dim, i);
          s->flag() = -11;
          s->v(0)   = 11.0f;
          s->v(1)   = 29.0f;
          s->v(2)   = 30.4f;
        }
      }
    });
    bi->flag() = si->flag();
    bi->v(0)   = si->v(0);
    bi->v(1)   = si->v(1);
    bi->v(2)   = si->v(2);
    __syncthreads();
        
    // Accumulate:
    ripple::unrolled_for<3>([&] (auto dim) {
      for (auto sign = -1; sign <= 1; sign += 2) {
        for (auto i = std::size_t{1}; i <= si.padding(); ++i) {
          auto s = si.offset(dim, sign * i);
          
          bi->flag() += s->flag();
          bi->v(0)   += s->v(0);
          bi->v(1)   += s->v(1);
          bi->v(2)   += s->v(2);
        }
      }
    });
  });

  const auto b      = b_dev.as_host();
  const float scale = 6 * padding + 1;
  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi  = b(i, j, k);
        const auto idx = static_cast<float>(i);

        EXPECT_NEAR(bi->flag(), -11 * scale  , tol);
        EXPECT_NEAR(bi->v(0)  , 11.0f * scale, tol);
        EXPECT_NEAR(bi->v(1)  , 29.0f * scale, tol);
        EXPECT_NEAR(bi->v(2)  , 30.4f * scale, tol);
      }
    }
  }
}

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_BLOCK_TESTS_HPP


