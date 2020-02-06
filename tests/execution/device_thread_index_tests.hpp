//==--- ../execution/device_thread_index_tests.hpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_thread_index_tests.hpp
/// \brief This file contains tests for thead indexing on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_EXECUTION_DEVICE_THREAD_INDEX_TESTS_HPP
#define RIPPLE_TESTS_EXECUTION_DEVICE_THREAD_INDEX_TESTS_HPP

#include <ripple/core/container/device_block.hpp>
#include <ripple/core/execution/thread_index.hpp>
#include <ripple/core/functional/invoke.hpp>
#include <gtest/gtest.h>

template <size_t Dims>
struct Index {
  ripple_host_device auto operator[](int i) -> size_t& {
    return _data[i];
  }

  ripple_host_device auto operator[](int i) const -> const size_t& {
    return _data[i];
  }
 private:
  size_t _data[Dims];
};

TEST(execution_thread_index, global_index_correct_1d) {
  ripple::device_block_1d_t<size_t> b(491);

  ripple::invoke(b, [] ripple_host_device (auto bi) {
    *bi = ripple::global_idx(ripple::dim_x);
  });

  auto hb = b.as_host();
  for (auto i : ripple::range(hb.size(ripple::dim_x))) {
    EXPECT_EQ(*hb(i), i);
  }
}

TEST(execution_thread_index, global_index_correct_2d) {
  ripple::device_block_2d_t<Index<2>> b(137, 62);
  ripple::invoke(b, [] ripple_host_device (auto bi) {
    (*bi)[0] = ripple::global_idx(ripple::dim_x);
    (*bi)[1] = ripple::global_idx(ripple::dim_y);
  });

  auto hb = b.as_host();
  for (auto j : ripple::range(hb.size(ripple::dim_y))) {
    for (auto i : ripple::range(hb.size(ripple::dim_x))) {
      const auto bi = hb(i, j);
      EXPECT_EQ((*bi)[0], i);
      EXPECT_EQ((*bi)[1], j);
    }
  }
}

TEST(execution_thread_index, global_index_correct_3d) {
  ripple::device_block_3d_t<Index<3>> b(141, 37, 22);

  ripple::invoke(b, [] ripple_host_device (auto bi) {
    (*bi)[0] = ripple::global_idx(ripple::dim_x);
    (*bi)[1] = ripple::global_idx(ripple::dim_y);
    (*bi)[2] = ripple::global_idx(ripple::dim_z);
  });

  auto hb = b.as_host();
  for (auto k : ripple::range(hb.size(ripple::dim_z))) {
    for (auto j : ripple::range(hb.size(ripple::dim_y))) {
      for (auto i : ripple::range(hb.size(ripple::dim_x))) {
        const auto bi = hb(i, j, k);

        EXPECT_EQ((*bi)[0], i);
        EXPECT_EQ((*bi)[1], j);
        EXPECT_EQ((*bi)[2], k);
      }
    }
  }
}

#endif // RIPPLE_TESTS_EXECUTION_DEVICE_THREAD_INDEX_TESTS_HPP

