//==--- ../execution/thread_index_tests.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  thread_index_tests.hpp
/// \brief This file contains tests for thead indexing on the host.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_EXECUTION_THREAD_INDEX_TESTS_HPP
#define RIPPLE_TESTS_EXECUTION_THREAD_INDEX_TESTS_HPP

#include <ripple/core/container/host_block.hpp>
#include <ripple/core/execution/thread_index.hpp>
#include <ripple/core/functional/invoke.hpp>
#include <gtest/gtest.h>
#include <array>

TEST(execution_thread_index, global_index_correct_1d) {
  using index1d_t = size_t;
  ripple::host_block_1d_t<index1d_t> b(491);

  ripple::invoke(b, [] (auto bi) {
    *bi = ripple::global_idx(ripple::dim_x);
  });

  for (auto i : ripple::range(b.size(ripple::dim_x))) {
    EXPECT_EQ(*b(i), i);
  }
}

TEST(execution_thread_index, global_index_correct_2d) {
  using index2d_t = std::array<size_t, 2>;
  ripple::host_block_2d_t<index2d_t> b(137, 62);

  ripple::invoke(b, [] (auto bi) {
    (*bi)[0] = ripple::global_idx(ripple::dim_x);
    (*bi)[1] = ripple::global_idx(ripple::dim_y);
  });

  for (auto j : ripple::range(b.size(ripple::dim_y))) {
    for (auto i : ripple::range(b.size(ripple::dim_x))) {
      const auto bi = b(i, j);
      EXPECT_EQ((*bi)[0], i);
      EXPECT_EQ((*bi)[1], j);
    }
  }
}

TEST(execution_thread_index, global_index_correct_3d) {
  using index3_t = std::array<size_t, 3>;
  ripple::host_block_3d_t<index3_t> b(141, 37, 22);

  ripple::invoke(b, [] (auto bi) {
    (*bi)[0] = ripple::global_idx(ripple::dim_x);
    (*bi)[1] = ripple::global_idx(ripple::dim_y);
    (*bi)[2] = ripple::global_idx(ripple::dim_z);
  });

  for (auto k : ripple::range(b.size(ripple::dim_z))) {
    for (auto j : ripple::range(b.size(ripple::dim_y))) {
      for (auto i : ripple::range(b.size(ripple::dim_x))) {
        const auto bi = b(i, j, k);

        EXPECT_EQ((*bi)[0], i);
        EXPECT_EQ((*bi)[1], j);
        EXPECT_EQ((*bi)[2], k);
      }
    }
  }
}

#endif // RIPPLE_TESTS_EXECUTION_THREAD_INDEX_TESTS_HPP

