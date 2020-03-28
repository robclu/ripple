//==--- ripple/core/tests/math/lerp_tests.hpp -------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  lerp_tests.hpp
/// \brief This file defines tests for linear interpolation.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_MATH_LERP_TESTS_HPP
#define RIPPLE_TESTS_MATH_LERP_TESTS_HPP

#include <ripple/core/container/vec.hpp>
#include <ripple/core/container/host_block.hpp>
#include <ripple/core/math/lerp.hpp>
#include <gtest/gtest.h>

TEST(math_tests, can_lerp_1d_non_udt) {
  using data_t   = double;
  using weight_t = ripple::vec_1d_t<data_t>;
  ripple::host_block_1d_t<data_t> b(3);

  *b(0) = data_t{1};
  *b(1) = data_t{2}; 
  *b(2) = data_t{3};

  // Test basic cases:
  auto r1 = ripple::math::lerp(b(1), weight_t{0.5});
  auto r2 = ripple::math::lerp(b(1), weight_t{-0.5});
  auto r3 = ripple::math::lerp(b(0), weight_t{1.25});
  auto r4 = ripple::math::lerp(b(2), weight_t{-1.25});

  EXPECT_EQ(r1, 2.5);
  EXPECT_EQ(r2, 1.5);
  EXPECT_EQ(r3, 2.25);
  EXPECT_EQ(r4, 1.75);

  // Test cases for numerical loss of significance:
  EXPECT_EQ(ripple::math::lerp(b(1), weight_t{0}), *b(1));
  EXPECT_EQ(ripple::math::lerp(b(1), weight_t{1}), *b(2));
}

// Test for 1d lerp with a user defined stidable type which owns its data.
TEST(math_tests, can_lerp_1d_stridable_contig_owned) {
  using namespace ripple;
  using data_t   = float;
  using type_t   = Vector<data_t, 3, contiguous_owned_t>;
  using owned_t  = Vector<data_t, 3, contiguous_owned_t>;
  using weight_t = ripple::vec_1d_t<data_t>;
  host_block_1d_t<type_t> b(3);

  *b(0) = owned_t{1, 4, 5};
  *b(1) = owned_t{2, 7, 4}; 
  *b(2) = owned_t{3, 9, 12};

  // Test basic cases:
  auto r1 = ripple::math::lerp(b(1), weight_t{0.5});
  EXPECT_EQ(r1[0], 2.5); EXPECT_EQ(r1[1], 8); EXPECT_EQ(r1[2], 8);
  r1 = ripple::math::lerp(b(1), weight_t{-0.5});
  EXPECT_EQ(r1[0], 1.5); EXPECT_EQ(r1[1], 5.5); EXPECT_EQ(r1[2], 4.5);
  r1 = ripple::math::lerp(b(0), weight_t{1.25});
  EXPECT_EQ(r1[0], 2.25); EXPECT_EQ(r1[1], 7.5); EXPECT_EQ(r1[2], 6);
  r1 = ripple::math::lerp(b(2), weight_t{-1.25});
  EXPECT_EQ(r1[0], 1.75); EXPECT_EQ(r1[1], 6.25); EXPECT_EQ(r1[2], 4.25);

  // Test edge cases:
  r1 = ripple::math::lerp(b(1), weight_t{0});
  EXPECT_EQ(r1[0], 2); EXPECT_EQ(r1[1], 7); EXPECT_EQ(r1[2], 4);
  r1 = ripple::math::lerp(b(1), weight_t{1});
  EXPECT_EQ(r1[0], 3); EXPECT_EQ(r1[1], 9); EXPECT_EQ(r1[2], 12);
}

// Test for 1d lerp with a user defined stidable type with contig view.
TEST(math_tests, can_lerp_1d_stridable_contig_view) {
  using namespace ripple;
  using data_t   = float;
  using type_t   = Vector<data_t, 3, contiguous_view_t>;
  using owned_t  = Vector<data_t, 3, contiguous_owned_t>;
  using weight_t = ripple::vec_1d_t<data_t>;
  host_block_1d_t<type_t> b(3);

  *b(0) = owned_t{1, 4, 5};
  *b(1) = owned_t{2, 7, 4}; 
  *b(2) = owned_t{3, 9, 12};

  // Test basic cases:
  auto r1 = ripple::math::lerp(b(1), weight_t{0.5});
  EXPECT_EQ(r1[0], 2.5); EXPECT_EQ(r1[1], 8); EXPECT_EQ(r1[2], 8);
  r1 = ripple::math::lerp(b(1), weight_t{-0.5});
  EXPECT_EQ(r1[0], 1.5); EXPECT_EQ(r1[1], 5.5); EXPECT_EQ(r1[2], 4.5);
  r1 = ripple::math::lerp(b(0), weight_t{1.25});
  EXPECT_EQ(r1[0], 2.25); EXPECT_EQ(r1[1], 7.5); EXPECT_EQ(r1[2], 6);
  r1 = ripple::math::lerp(b(2), weight_t{-1.25});
  EXPECT_EQ(r1[0], 1.75); EXPECT_EQ(r1[1], 6.25); EXPECT_EQ(r1[2], 4.25);

  // Test edge cases:
  r1 = ripple::math::lerp(b(1), weight_t{0});
  EXPECT_EQ(r1[0], 2); EXPECT_EQ(r1[1], 7); EXPECT_EQ(r1[2], 4);
  r1 = ripple::math::lerp(b(1), weight_t{1});
  EXPECT_EQ(r1[0], 3); EXPECT_EQ(r1[1], 9); EXPECT_EQ(r1[2], 12);
}

// Test for 1d lerp with a user defined stidable type with strided view.
TEST(math_tests, can_lerp_1d_stridable_view) {
  using namespace ripple;
  using data_t   = float;
  using type_t   = Vector<data_t, 3, strided_view_t>;
  using owned_t  = Vector<data_t, 3, contiguous_owned_t>;
  using weight_t = ripple::vec_1d_t<data_t>;
  host_block_1d_t<type_t> b(3);

  *b(0) = owned_t{1, 4, 5};
  *b(1) = owned_t{2, 7, 4}; 
  *b(2) = owned_t{3, 9, 12};

  // Test basic cases:
  auto r1 = ripple::math::lerp(b(1), weight_t{0.5});
  EXPECT_EQ(r1[0], 2.5); EXPECT_EQ(r1[1], 8); EXPECT_EQ(r1[2], 8);
  r1 = ripple::math::lerp(b(1), weight_t{-0.5});
  EXPECT_EQ(r1[0], 1.5); EXPECT_EQ(r1[1], 5.5); EXPECT_EQ(r1[2], 4.5);
  r1 = ripple::math::lerp(b(0), weight_t{1.25});
  EXPECT_EQ(r1[0], 2.25); EXPECT_EQ(r1[1], 7.5); EXPECT_EQ(r1[2], 6);
  r1 = ripple::math::lerp(b(2), weight_t{-1.25});
  EXPECT_EQ(r1[0], 1.75); EXPECT_EQ(r1[1], 6.25); EXPECT_EQ(r1[2], 4.25);

  // Test edge cases:
  r1 = ripple::math::lerp(b(1), weight_t{0});
  EXPECT_EQ(r1[0], 2); EXPECT_EQ(r1[1], 7); EXPECT_EQ(r1[2], 4);
  r1 = ripple::math::lerp(b(1), weight_t{1});
  EXPECT_EQ(r1[0], 3); EXPECT_EQ(r1[1], 9); EXPECT_EQ(r1[2], 12);
}

#endif // RIPPLE_TESTS_MATH_LERP_TESTS_HPP

