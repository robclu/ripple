//==--- ripple/core/tests/boundary/device_load_boundary_tests.hpp -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_load_boundary_tests.hpp
/// \brief This file defines tests for device boundary loading.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_DEVICE_LOAD_BOUNDARY_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_DEVICE_LOAD_BOUNDARY_TESTS_HPP

#include <ripple/core/boundary/load_boundary.hpp>
#include <ripple/core/boundary/boundary_loader.hpp>
#include <ripple/core/boundary/fo_extrap_loader.hpp>
#include <ripple/core/container/device_block.hpp>
#include <ripple/core/functional/invoke.hpp>
#include <gtest/gtest.h>

//==--- [NOTE] -------------------------------------------------------------==//
//
// Currently, the implemenatation requires that the maximum padding value for a
// single side of a dimension is half the dimension size. In almost all cases
// this is not a problem, and in most cases is actually beneficial because it
// ensures that the ratio of data cells to padding cells is high. However, for
// 2D, it may be a problem for wide stencils (i.e 9 x 9 x 9), which will require
// a padding size of 4, hence requiring that the minimum block size for shared
// memory use would be 8 x 8 x 8, which is possible on all architectures.
//
//==------------------------------------------------------------------------==//

// Test loader class for numeric types.
template <typename T>
struct NumericLoader : public ripple::BoundaryLoader<NumericLoader<T>> {
  using value_t = std::decay_t<T>;

  //==--- [any dim] --------------------------------------------------------==//
  
  // Loads the front data in the dim dimension.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto load_front(
    Iterator&& it, int index, Dim&& dim
  ) const -> void {
    *it = std::max(abs_index(index) * value_t(10), *it.offset(dim, index));
  }

  // Loads the back data in the dim dimension.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto load_back(
    Iterator&& it, int index, Dim&& dim
  ) const -> void {
    *it = std::max(abs_index(index) * value_t(10), *it.offset(dim, index));
  }

 private:
  // returns the abs value of the index.
  ripple_host_device auto abs_index(int index) const -> T {
    return static_cast<value_t>(ripple::math::sign(index) * index);
  }
};

// Test loader class for loading the padding as constant, so that testing is
// made easier:
template <typename T>
struct ConstantLoader : public ripple::BoundaryLoader<ConstantLoader<T>> {
  using value_t = std::decay_t<T>;

  // Defines the value that is loaded:
  static constexpr auto value = value_t{1};

  //==--- [any dim] --------------------------------------------------------==//
  
  // Loads the front data in the dim dimension.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto load_front(
    Iterator&& it, int index, Dim&& dim
  ) const -> void {
    *it = value;
  }

  // Loads the back data in the dim dimension.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto load_back(
    Iterator&& it, int index, Dim&& dim
  ) const -> void {
    *it = value;
  }
};

//==--- [1st order extrapolation] ------------------------------------------==//

TEST(boundary_device_load_boundary, can_load_boundar_fo_exrap_3d) {
  using data_t = float;
  constexpr std::size_t padding = 1;
  constexpr std::size_t size_x  = 17;
  constexpr std::size_t size_y  = 32;
  constexpr std::size_t size_z  = 24;

  ripple::device_block_3d_t<data_t> block(padding, size_x, size_y, size_z);
  ripple::FOExtrapLoader loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{10};
  });

  // Load boundaries
  ripple::load_global_boundary(block, loader);

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  // Move iterator into the padding area:
  it.shift(ripple::dim_x, -1 * static_cast<int>(it.padding()));
  it.shift(ripple::dim_y, -1 * static_cast<int>(it.padding()));
  it.shift(ripple::dim_z, -1 * static_cast<int>(it.padding()));

  const auto elements_x = hblock.size(ripple::dim_x) + 2 * hblock.padding();
  const auto elements_y = hblock.size(ripple::dim_y) + 2 * hblock.padding();
  const auto elements_z = hblock.size(ripple::dim_z) + 2 * hblock.padding();
  for (auto k = 0; k < elements_z; ++k) {
    for (auto j = 0; j < elements_y; ++j) {
      for (auto i = 0; i < elements_x; ++i) {
        sum += *(it
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k)
        );
      }
    }
  }
 
  const auto pad = 2 * padding; 
  auto total = data_t{10} * (size_x + pad) * (size_y + pad) * (size_z + pad);
  EXPECT_EQ(total, sum);
}

//==--- [other loading] ----------------------------------------------------==//

TEST(boundary_device_load_boundary, can_load_boundary_1d) {
  // Part 1: Test loading of global data:
  using data_t = float;
  constexpr std::size_t padding = 3;
  constexpr std::size_t size_x  = 200;

  ripple::device_block_1d_t<data_t> block(padding, size_x);
  NumericLoader<data_t> loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = 10.0f;
  });

  // Load boundaries
  ripple::load_global_boundary(block, loader);
  
  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  // Move iterator into the padding area:
  it.shift(ripple::dim_x, -1 * static_cast<int>(it.padding()));

  const auto elements = hblock.size(ripple::dim_x) + 2 * hblock.padding();
  for (auto i = 0; i < elements; ++i) {
    sum += *(it.offset(ripple::dim_x, i));
  }

  auto total = 0.0f;
  for (auto i = 1; i <= hblock.padding(); ++i) {
    total += 10.0f * i * 2.0f;
  }
  total += hblock.size(ripple::dim_x) * 10.0f;

  EXPECT_EQ(total, sum);
}


TEST(boundary_device_load_boundary, can_load_internal_1d) {
  using data_t   = float;
  using loader_t = ConstantLoader<data_t>;
  constexpr std::size_t padding = 3;
  constexpr std::size_t size_x  = 2000;

  ripple::device_block_1d_t<data_t> block(padding, size_x);
  loader_t loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = loader_t::value;
  });

  // Load boundaries
  ripple::load_global_boundary(block, loader);

  // Sum from values around each cell:
  using exec_params_t = ripple::StaticExecParams<256, 1, 1, padding, data_t>;
  ripple::invoke(block, exec_params_t(), 
    [] ripple_host_device (auto it, auto shared_it) {
      *shared_it = *it;
      __syncthreads();

      for (auto i = 1; i <= shared_it.padding(); ++i) {
        *it += *shared_it.offset(ripple::dim_x, -i) +
               *shared_it.offset(ripple::dim_x,  i);
      }
    }
  );

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  // Move iterator into the padding area:
  it.shift(ripple::dim_x, -1 * static_cast<int>(it.padding()));

  const auto elements = hblock.size(ripple::dim_x) + 2 * hblock.padding();
  for (auto i = 0; i < elements; ++i) {
    sum += *(it.offset(ripple::dim_x, i));
  }

  // Sum of padding values and the internally accumulated values:
  auto total = loader_t::value * padding * 2.0f
    + hblock.size(ripple::dim_x) * (loader_t::value * (2 * padding + 1));

  EXPECT_EQ(total, sum);
}

//==--- [2d] ---------------------------------------------------------------==//

// This test loads the data as increading values of 10 in the paddings layers,
// so for a 4,4 square with 2 padding layers, the data would look like:
//
// ~~~
//  -----------------------------
//  | 20 20 |20 20 20 20| 20 20 |
//  | 20 10 |10 10 10 10 |10 20 |
//  |---------------------------|
//  | 20 10 |10 10 10 10| 10 20 |
//  | 20 10 |10 10 10 10| 10 20 |
//  | 20 10 |10 10 10 10| 10 20 |
//  | 20 10 |10 10 10 10| 10 20 |
//  | ------------------- ------|
//  | 20 10 |10 10 10 10| 10 20 |
//  | 20 20 |20 20 20 20| 20 20 |
//  -----------------------------
//~~~
TEST(boundary_device_load_boundary, can_load_boundary_2d_small) {
  using data_t = float;
  constexpr std::size_t padding = 1;
  constexpr std::size_t size_x  = 2;
  constexpr std::size_t size_y  = 2;

  ripple::device_block_2d_t<data_t> block(padding, size_x, size_y);
  NumericLoader<data_t> loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{10};
  });

  // Load boundaries
  ripple::load_global_boundary(block, loader);

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  // Move iterator into the padding area:
  it.shift(ripple::dim_x, -1 * static_cast<int>(it.padding()));
  it.shift(ripple::dim_y, -1 * static_cast<int>(it.padding()));

  const auto elements_x = hblock.size(ripple::dim_x) + 2 * hblock.padding();
  const auto elements_y = hblock.size(ripple::dim_y) + 2 * hblock.padding();
  for (auto j = 0; j < elements_y; ++j) {
    for (auto i = 0; i < elements_x; ++i) {
      sum += *(it.offset(ripple::dim_x, i).offset(ripple::dim_y, j));
    }
  }

  auto total = data_t{0};
  for (auto i = 0; i < hblock.padding(); ++i) {
    const auto gv       = (i + 1) * data_t{10};
    const auto x        = 2 * (size_x + 2 * i) * gv;
    const auto y        = 2 * (size_y + 2 * i) * gv;
    const auto corners  = 4 * gv;
    total += x + y + corners;
  }
  total += size_x * size_y * data_t{10};

  EXPECT_EQ(total, sum);
}

// Large version of above test
TEST(boundary_device_load_boundary, can_load_boundary_2d_large) {
  using data_t = float;
  constexpr std::size_t padding = 3;
  constexpr std::size_t size_x  = 200;
  constexpr std::size_t size_y  = 200;

  ripple::device_block_2d_t<data_t> block(padding, size_x, size_y);
  NumericLoader<data_t> loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{10};
  });

  // Load boundaries
  ripple::load_global_boundary(block, loader);

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  // Move iterator into the padding area:
  it.shift(ripple::dim_x, -1 * static_cast<int>(it.padding()));
  it.shift(ripple::dim_y, -1 * static_cast<int>(it.padding()));

  const auto elements_x = hblock.size(ripple::dim_x) + 2 * hblock.padding();
  const auto elements_y = hblock.size(ripple::dim_y) + 2 * hblock.padding();
  for (auto j = 0; j < elements_y; ++j) {
    for (auto i = 0; i < elements_x; ++i) {
      sum += *(it.offset(ripple::dim_x, i).offset(ripple::dim_y, j));
    }
  }

  auto total = data_t{0};
  for (auto i = 0; i < hblock.padding(); ++i) {
    const auto gv       = (i + 1) * data_t{10};
    const auto x        = 2 * (size_x + 2 * i) * gv;
    const auto y        = 2 * (size_y + 2 * i) * gv;
    const auto corners  = 4 * gv;
    total += x + y + corners;
  }
  total += size_x * size_y * data_t{10};

  EXPECT_EQ(total, sum);
}

// Test for internal loading of data (i.e shared memory). This is a small test
// case, which tests that the shared memory is loaded correctly for blocks which
// are smaller than the execution block size.
TEST(boundary_device_load_boundary, can_load_internal_2d_small) {
  using data_t   = float;
  using loader_t = ConstantLoader<data_t>;
  constexpr std::size_t padding = 2;
  constexpr std::size_t size_x  = 4;
  constexpr std::size_t size_y  = 5;

  ripple::device_block_2d_t<data_t> block(padding, size_x, size_y);
  loader_t loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = loader_t::value;;
  });

  ripple::load_global_boundary(block, loader);

  // Set each cell to the sum of the padding x padding block in which the cell
  // is centerd:
  using exec_params_t = ripple::StaticExecParams<16, 16, 1, padding, data_t>;
  ripple::invoke(block, exec_params_t(), 
    [] ripple_host_device (auto it, auto shared_it) {
      const int neg_padding = -1 * static_cast<int>(shared_it.padding());
      shared_it.shift(ripple::dim_x, neg_padding);
      shared_it.shift(ripple::dim_y, neg_padding);
      for (auto j : ripple::range(2 * shared_it.padding() + 1)) {
        for (auto i : ripple::range(2 * shared_it.padding() + 1)) {
          *it += *shared_it.offset(ripple::dim_x, i);
        }
        shared_it.shift(ripple::dim_y, 1);
      }
    }
  );

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  const auto elements_x = hblock.size(ripple::dim_x);
  const auto elements_y = hblock.size(ripple::dim_y);

  for (auto j = 0; j < elements_y; ++j) {
    for (auto i = 0; i < elements_x; ++i) {
      sum += *(it.offset(ripple::dim_x, i).offset(ripple::dim_y, j));
    }
  }

  const auto elems_in_sum = (2 * padding + 1) * (2 * padding + 1) + 1;
  const auto total = elements_x * elements_y * elems_in_sum * loader_t::value;

  EXPECT_EQ(total, sum);
}

// Test for internal loading of data (i.e shared memory). This is a small test
// case, which tests that the shared memory is loaded correctly for blocks which
// are smaller than the execution block size.
TEST(boundary_device_load_boundary, can_load_internal_3d_small) {
  using data_t   = float;
  using loader_t = ConstantLoader<data_t>;
  constexpr std::size_t padding = 2;
  constexpr std::size_t size_x  = 5;
  constexpr std::size_t size_y  = 5;
  constexpr std::size_t size_z  = 7;

  ripple::device_block_3d_t<data_t> block(padding, size_x, size_y, size_z);
  loader_t loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = loader_t::value;;
  });

  ripple::load_global_boundary(block, loader);

  // Set each cell to the sum of the padding x padding block in which the cell
  // is centerd:
  using exec_params_t = ripple::StaticExecParams<8, 8, 8, padding, data_t>;
  ripple::invoke(block, exec_params_t(), 
    [] ripple_host_device (auto it, auto shared_it) {
      const int neg_padding = -1 * static_cast<int>(shared_it.padding());
      shared_it.shift(ripple::dim_x, neg_padding);
      shared_it.shift(ripple::dim_y, neg_padding);
      shared_it.shift(ripple::dim_z, neg_padding);
      for (auto k : ripple::range(2 * shared_it.padding() + 1)) {
        for (auto j : ripple::range(2 * shared_it.padding() + 1)) {
          for (auto i : ripple::range(2 * shared_it.padding() + 1)) {
            *it += *shared_it
              .offset(ripple::dim_x, i)
              .offset(ripple::dim_y, j)
              .offset(ripple::dim_z, k);
          }
        }
      }
    }
  );

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  const auto elements_x = hblock.size(ripple::dim_x);
  const auto elements_y = hblock.size(ripple::dim_y);
  const auto elements_z = hblock.size(ripple::dim_z);
  for (auto k = 0; k < elements_z; ++k) {
    for (auto j = 0; j < elements_y; ++j) {
      for (auto i = 0; i < elements_x; ++i) {
        sum += *(it
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k)
        );
      }
    }
  }

  const auto length       = 2 * padding + 1;
  const auto elems_in_sum = length * length * length + 1;
  const auto total = 
    elements_x * elements_y * elements_z * elems_in_sum * loader_t::value;

  EXPECT_EQ(total, sum);
}

TEST(boundary_device_load_boundary, can_load_boundary_3d_large) {
  using data_t = float;
  constexpr std::size_t padding = 1;
  constexpr std::size_t size_x  = 17;
  constexpr std::size_t size_y  = 32;
  constexpr std::size_t size_z  = 24;

  ripple::device_block_3d_t<data_t> block(padding, size_x, size_y, size_z);
  NumericLoader<data_t> loader;

  // Fill internal data to const value:
  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{10};
  });

  // Load boundaries
  ripple::load_global_boundary(block, loader);

  auto hblock = block.as_host();
  auto it     = hblock.begin();
  auto sum    = data_t{0};

  // Move iterator into the padding area:
  it.shift(ripple::dim_x, -1 * static_cast<int>(it.padding()));
  it.shift(ripple::dim_y, -1 * static_cast<int>(it.padding()));
  it.shift(ripple::dim_z, -1 * static_cast<int>(it.padding()));

  const auto elements_x = hblock.size(ripple::dim_x) + 2 * hblock.padding();
  const auto elements_y = hblock.size(ripple::dim_y) + 2 * hblock.padding();
  const auto elements_z = hblock.size(ripple::dim_z) + 2 * hblock.padding();
  for (auto k = 0; k < elements_z; ++k) {
    for (auto j = 0; j < elements_y; ++j) {
      for (auto i = 0; i < elements_x; ++i) {
        sum += *(it
          .offset(ripple::dim_x, i)
          .offset(ripple::dim_y, j)
          .offset(ripple::dim_z, k)
        );
      }
    }
  }
  
  const auto size = [] (auto pad) {
    return (size_x + pad) * (size_y + pad) * (size_z + pad);
  };
  auto total = data_t{0};
  for (auto i = 1; i <= hblock.padding(); ++i) {
    const auto gv       = i * data_t{10};
    const auto pad_i    = 2 * i;
    const auto pad_ip   = 2 * (i - 1);

    // Big cube minus smaller cube, gives number of outer cells:
    const auto cells = size(pad_i) - size(pad_ip);
    total += cells * gv;
  }
  total += size(0) * data_t{10};

  EXPECT_EQ(total, sum);
}

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_LOAD_BOUNDARY_TESTS_HPP
