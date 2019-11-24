//==--- ripple/tests/boundary/device_load_boundary_tests.hpp -*- C++ -*- ---==//
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

#include <ripple/boundary/load_boundary.hpp>
#include <ripple/boundary/boundary_loader.hpp>
#include <ripple/container/device_block.hpp>
#include <ripple/functional/invoke.hpp>
#include <gtest/gtest.h>

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

TEST(boundary_device_load_boundary, can_load_boundary_1d) {
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

TEST(boundary_device_load_boundary, can_load_boundary_2d) {
  using data_t = float;
  constexpr std::size_t padding = 2;
  constexpr std::size_t size_x  = 8;
  constexpr std::size_t size_y  = 8;

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

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_LOAD_BOUNDARY_TESTS_HPP
