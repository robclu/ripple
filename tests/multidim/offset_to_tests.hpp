//==--- ripple/core/tests/multidim/offset_to_tests.hpp ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  offset_to_tests.hpp
/// \brief This file contains tests for computing the offset to an element in a
///        multidimensional space.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_MULTIDIM_OFFSET_TO_TESTS_HPP
#define RIPPLE_TESTS_MULTIDIM_OFFSET_TO_TESTS_HPP

#include <ripple/core/multidim/offset_to.hpp>

constexpr auto size_x         = std::size_t{20};
constexpr auto size_y         = std::size_t{30};
constexpr auto size_z         = std::size_t{40};
constexpr auto arr_start_size = std::size_t{2};
constexpr auto arr_end_size   = std::size_t{6};
constexpr auto arr_step_size  = std::size_t{1};

auto arr_size_range() {
  return ripple::range(arr_start_size, arr_end_size, arr_step_size);
}

TEST(multidim_offset_to, correct_offset_contig_elements_1d) {
  using space_t    = ripple::DynamicMultidimSpace<1>;
  const auto space = space_t{size_x}; 

  for (auto i : ripple::range(size_x)) {
    EXPECT_EQ(ripple::offset_to_soa(space, 1, i), i);
    EXPECT_EQ(ripple::offset_to_aos(space, 1, i), i);
  }
}

TEST(multidim_offset_to, correct_offset_soa_elements_1d) {
  using space_t             = ripple::DynamicMultidimSpace<1>;
  const auto space          = space_t{size_x};

  for (auto soa_size : arr_size_range())
    for (auto i : ripple::range(size_x)) {
      EXPECT_EQ(ripple::offset_to_soa(space, soa_size, i), i);
    }
}

TEST(multidim_offset_to, correct_offset_aos_elements_1d) {
  using space_t       = ripple::DynamicMultidimSpace<1>;
  const auto space    = space_t{size_x};

  for (auto aos_size : arr_size_range()) {
    for (auto i : ripple::range(size_x)) {
      EXPECT_EQ(ripple::offset_to_aos(space, aos_size, i), i * aos_size);
    }
  }
}

TEST(multidim_offset_to, correct_offset_contig_elements_2d) {
  using space_t    = ripple::DynamicMultidimSpace<2>;
  const auto space = space_t{size_x, size_y}; 

  for (auto j : ripple::range(size_y)) {
    for (auto i : ripple::range(size_x)) {
      EXPECT_EQ(ripple::offset_to_soa(space, 1, i, j), i + j * size_x);
      EXPECT_EQ(ripple::offset_to_aos(space, 1, i, j), i + j * size_x);
    }
  }
}

TEST(multidim_offset_to, correct_offset_soa_elements_2d) {
  using space_t    = ripple::DynamicMultidimSpace<2>;
  const auto space = space_t{size_x, size_y}; 

  for (auto soa_size : arr_size_range()) {
    for (auto j : ripple::range(size_y)) {
      for (auto i : ripple::range(size_x)) {
        const auto offset = i + j * size_x * soa_size;
        EXPECT_EQ(ripple::offset_to_soa(space, soa_size, i, j), offset);
      }
    }
  }
}

TEST(multidim_offset_to, correct_offset_aos_elements_2d) {
  using space_t    = ripple::DynamicMultidimSpace<2>;
  const auto space = space_t{size_x, size_y}; 

  for (auto aos_size : arr_size_range()) {
    for (auto j : ripple::range(size_y)) {
      for (auto i : ripple::range(size_x)) {
        const auto offset = i * aos_size + j * size_x * aos_size;
        EXPECT_EQ(ripple::offset_to_aos(space, aos_size, i, j), offset);
      }
    }
  }
}

TEST(multidim_offset_to, correct_offset_contig_elements_3d) {
  using space_t    = ripple::DynamicMultidimSpace<3>;
  const auto space = space_t{size_x, size_y, size_z}; 

  for (auto k : ripple::range(size_z)) {
    for (auto j : ripple::range(size_y)) {
      for (auto i : ripple::range(size_x)) {
        const auto offset = i + j * size_x + k * size_x * size_y;
        EXPECT_EQ(ripple::offset_to_soa(space, 1, i, j, k), offset);
        EXPECT_EQ(ripple::offset_to_aos(space, 1, i, j, k), offset);
      }
    }
  }
}

TEST(multidim_offset_to, correct_offset_soa_elements_3d) {
  using space_t    = ripple::DynamicMultidimSpace<3>;
  const auto space = space_t{size_x, size_y, size_z}; 

  for (auto soa_size : arr_size_range()) {
    for (auto k : ripple::range(size_z)) {
      for (auto j : ripple::range(size_y)) {
        for (auto i : ripple::range(size_x)) {
          const auto offset = 
            i + j * size_x * soa_size + k * size_x * size_y * soa_size;
          EXPECT_EQ(ripple::offset_to_soa(space, soa_size, i, j, k), offset);
        }
      }
    }
  }
}

TEST(multidim_offset_to, correct_offset_aos_elements_3d) {
  using space_t    = ripple::DynamicMultidimSpace<3>;
  const auto space = space_t{size_x, size_y, size_z}; 

  for (auto aos_size : arr_size_range()) {
    for (auto k : ripple::range(size_z)) {
      for (auto j : ripple::range(size_y)) {
        for (auto i : ripple::range(size_x)) {
          const auto offset = 
            i * aos_size + 
            j * aos_size * size_x + 
            k * aos_size * size_x * size_y;

          EXPECT_EQ(ripple::offset_to_aos(space, aos_size, i, j, k), offset);
        }
      }
    }
  }
}

#endif // RIPPLE_TESTS_MULTIDIM_OFFSET_TO_TESTS_HPP
