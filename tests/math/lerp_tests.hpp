//==--- ripple/tests/math/lerp_tests.hpp ------------------- -*- C++ -*- ---==//
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

static constexpr double tol = 1e-6;

//==--- [1d tests] ---------------------------------------------------------==//

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

//==--- [2d tests] ---------------------------------------------------------==//

TEST(math_tests, can_lerp_2d_non_udt) {
  using data_t   = double;
  using weight_t = ripple::vec_2d_t<data_t>;
  ripple::host_block_2d_t<data_t> b(3, 3);

  *b(0, 0) = 1; *b(1, 0) = 2; *b(2, 0) = 3;
  *b(0, 1) = 4; *b(1, 1) = 5; *b(2, 1) = 6;
  *b(0, 2) = 7; *b(1, 2) = 8; *b(2, 2) = 9;

  auto c = b(1, 1);

  // Test basic cases:
  auto r = ripple::math::lerp(c, weight_t{0.5, 0.5});
  EXPECT_EQ(r, 7);
  r = ripple::math::lerp(c, weight_t{0.0, -1.0});
  EXPECT_EQ(r, 2.0);
  r = ripple::math::lerp(c, weight_t{-0.2, -0.8});
  EXPECT_NEAR(r, 2.4, tol);
  r = ripple::math::lerp(c, weight_t{-0.2, 0.8});
  EXPECT_NEAR(r, 7.2, tol);
  r = ripple::math::lerp(c, weight_t{0.2, -0.8});
  EXPECT_NEAR(r, 2.8, tol);
}

// Test for 2d lerp with a user defined stidable type which owns its data.
TEST(math_tests, can_lerp_2d_stridable_contig_owned) {
  using namespace ripple;
  using data_t   = float;
  using type_t   = Vector<data_t, 2, contiguous_owned_t>;
  using owned_t  = Vector<data_t, 2, contiguous_owned_t>;
  using weight_t = ripple::vec_2d_t<data_t>;
  host_block_2d_t<type_t> b(3, 3);

  *b(0, 0) = owned_t{1, 1}; *b(1, 0) = owned_t{2, 2}; *b(2, 0) = owned_t{3, 3};
  *b(0, 1) = owned_t{4, 4}; *b(1, 1) = owned_t{5, 5}; *b(2, 1) = owned_t{6, 6};
  *b(0, 2) = owned_t{7, 7}; *b(1, 2) = owned_t{8, 8}; *b(2, 2) = owned_t{9, 9};

  // Right and down:
  auto r = ripple::math::lerp(b(1, 1), weight_t{0.5, 0.5});
  EXPECT_EQ(r[0], data_t{7}); EXPECT_EQ(r[1], data_t{7}); 

  // Right and up:
  r = ripple::math::lerp(b(1, 1), weight_t{0.2, -0.8});
  EXPECT_NEAR(r[0], data_t{2.8}, tol); EXPECT_NEAR(r[1], data_t{2.8}, tol);

  // Left and down:
  r = ripple::math::lerp(b(1, 1), weight_t{-0.2, 0.8});
  EXPECT_NEAR(r[0], data_t{7.2}, tol); EXPECT_NEAR(r[1], data_t{7.2}, tol);

  // Left and up:
  r = ripple::math::lerp(b(1, 1), weight_t{-0.2, -0.8});
  EXPECT_NEAR(r[0], data_t{2.4}, tol); EXPECT_NEAR(r[1], data_t{2.4}, tol);

  // Test for zero, zero:
  r = ripple::math::lerp(b(1, 1), weight_t{0, 0});
  EXPECT_EQ(r[0], data_t{5}); EXPECT_EQ(r[1], data_t{5});

  // Test for zero, 1:
  r = ripple::math::lerp(b(1, 1), weight_t{0, 1});
  EXPECT_EQ(r[0], data_t{8}); EXPECT_EQ(r[1], data_t{8});

  // Test for 1, 0:
  r = ripple::math::lerp(b(1, 1), weight_t{1, 0});
  EXPECT_EQ(r[0], data_t{6}); EXPECT_EQ(r[1], data_t{6});

  // Test for 1, 1:
  r = ripple::math::lerp(b(1, 1), weight_t{1, 1});
  EXPECT_EQ(r[0], data_t{9}); EXPECT_EQ(r[1], data_t{9});
}

// Test for 2d lerp with a user defined stidable type which doesn't own its
// data.
TEST(math_tests, can_lerp_2d_stridable_contig_view) {
  using namespace ripple;
  using data_t   = float;
  using type_t   = Vector<data_t, 2, contiguous_view_t>;
  using owned_t  = Vector<data_t, 2, contiguous_owned_t>;
  using weight_t = ripple::vec_2d_t<data_t>;
  host_block_2d_t<type_t> b(3, 3);

  *b(0, 0) = owned_t{1, 1}; *b(1, 0) = owned_t{2, 2}; *b(2, 0) = owned_t{3, 3};
  *b(0, 1) = owned_t{4, 4}; *b(1, 1) = owned_t{5, 5}; *b(2, 1) = owned_t{6, 6};
  *b(0, 2) = owned_t{7, 7}; *b(1, 2) = owned_t{8, 8}; *b(2, 2) = owned_t{9, 9};

  // Right and down:
  auto r = ripple::math::lerp(b(1, 1), weight_t{0.5, 0.5});
  EXPECT_EQ(r[0], data_t{7}); EXPECT_EQ(r[1], data_t{7}); 

  // Right and up:
  r = ripple::math::lerp(b(1, 1), weight_t{0.2, -0.8});
  EXPECT_NEAR(r[0], data_t{2.8}, tol); EXPECT_NEAR(r[1], data_t{2.8}, tol);

  // Left and down:
  r = ripple::math::lerp(b(1, 1), weight_t{-0.2, 0.8});
  EXPECT_NEAR(r[0], data_t{7.2}, tol); EXPECT_NEAR(r[1], data_t{7.2}, tol);

  // Left and up:
  r = ripple::math::lerp(b(1, 1), weight_t{-0.2, -0.8});
  EXPECT_NEAR(r[0], data_t{2.4}, tol); EXPECT_NEAR(r[1], data_t{2.4}, tol);

  // Test for zero, zero:
  r = ripple::math::lerp(b(1, 1), weight_t{0, 0});
  EXPECT_EQ(r[0], data_t{5}); EXPECT_EQ(r[1], data_t{5});

  // Test for zero, 1:
  r = ripple::math::lerp(b(1, 1), weight_t{0, 1});
  EXPECT_EQ(r[0], data_t{8}); EXPECT_EQ(r[1], data_t{8});

  // Test for 1, 0:
  r = ripple::math::lerp(b(1, 1), weight_t{1, 0});
  EXPECT_EQ(r[0], data_t{6}); EXPECT_EQ(r[1], data_t{6});

  // Test for 1, 1:
  r = ripple::math::lerp(b(1, 1), weight_t{1, 1});
  EXPECT_EQ(r[0], data_t{9}); EXPECT_EQ(r[1], data_t{9});
}

// Test for 2d lerp with a user defined stidable type which doesn't own its
// data.
TEST(math_tests, can_lerp_2d_stridable_strided_view) {
  using namespace ripple;
  using data_t   = float;
  using type_t   = Vector<data_t, 2, strided_view_t>;
  using owned_t  = Vector<data_t, 2, contiguous_owned_t>;
  using weight_t = ripple::vec_2d_t<data_t>;
  host_block_2d_t<type_t> b(3, 3);

  *b(0, 0) = owned_t{1, 1}; *b(1, 0) = owned_t{2, 2}; *b(2, 0) = owned_t{3, 3};
  *b(0, 1) = owned_t{4, 4}; *b(1, 1) = owned_t{5, 5}; *b(2, 1) = owned_t{6, 6};
  *b(0, 2) = owned_t{7, 7}; *b(1, 2) = owned_t{8, 8}; *b(2, 2) = owned_t{9, 9};

  // Right and down:
  auto r = ripple::math::lerp(b(1, 1), weight_t{0.5, 0.5});
  EXPECT_EQ(r[0], data_t{7}); EXPECT_EQ(r[1], data_t{7}); 

  // Right and up:
  r = ripple::math::lerp(b(1, 1), weight_t{0.2, -0.8});
  EXPECT_NEAR(r[0], data_t{2.8}, tol); EXPECT_NEAR(r[1], data_t{2.8}, tol);

  // Left and down:
  r = ripple::math::lerp(b(1, 1), weight_t{-0.2, 0.8});
  EXPECT_NEAR(r[0], data_t{7.2}, tol); EXPECT_NEAR(r[1], data_t{7.2}, tol);

  // Left and up:
  r = ripple::math::lerp(b(1, 1), weight_t{-0.2, -0.8});
  EXPECT_NEAR(r[0], data_t{2.4}, tol); EXPECT_NEAR(r[1], data_t{2.4}, tol);

  // Test for zero, zero:
  r = ripple::math::lerp(b(1, 1), weight_t{0, 0});
  EXPECT_EQ(r[0], data_t{5}); EXPECT_EQ(r[1], data_t{5});

  // Test for zero, 1:
  r = ripple::math::lerp(b(1, 1), weight_t{0, 1});
  EXPECT_EQ(r[0], data_t{8}); EXPECT_EQ(r[1], data_t{8});

  // Test for 1, 0:
  r = ripple::math::lerp(b(1, 1), weight_t{1, 0});
  EXPECT_EQ(r[0], data_t{6}); EXPECT_EQ(r[1], data_t{6});

  // Test for 1, 1:
  r = ripple::math::lerp(b(1, 1), weight_t{1, 1});
  EXPECT_EQ(r[0], data_t{9}); EXPECT_EQ(r[1], data_t{9});
}

//==--- [3d tests] ---------------------------------------------------------==//

TEST(math_tests, can_lerp_3d_non_udt) {
  using data_t   = double;
  using weight_t = ripple::vec_3d_t<data_t>;
  ripple::host_block_3d_t<data_t> b(3, 3, 3);

  // First plane:
  *b(0, 0, 0) = 1; *b(1, 0, 0) = 2; *b(2, 0, 0) = 3;
  *b(0, 1, 0) = 4; *b(1, 1, 0) = 5; *b(2, 1, 0) = 6;
  *b(0, 2, 0) = 7; *b(1, 2, 0) = 8; *b(2, 2, 0) = 9;
  // Second plane:
  *b(0, 0, 1) = 9; *b(1, 0, 1) = 8; *b(2, 0, 1) = 7;
  *b(0, 1, 1) = 6; *b(1, 1, 1) = 5; *b(2, 1, 1) = 4;
  *b(0, 2, 1) = 3; *b(1, 2, 1) = 2; *b(2, 2, 1) = 1;
  // Third plane:
  *b(0, 0, 2) = 1; *b(1, 0, 2) = 2; *b(2, 0, 2) = 3;
  *b(0, 1, 2) = 4; *b(1, 1, 2) = 5; *b(2, 1, 2) = 6;
  *b(0, 2, 2) = 7; *b(1, 2, 2) = 8; *b(2, 2, 2) = 9;

  auto c = b(1, 1, 1);

  // Test simple case:
  auto r = ripple::math::lerp(c, weight_t{0.5, 0.5, 0.5});
  EXPECT_EQ(r, 5);

  // Right, down, forward:
  r = ripple::math::lerp(c, weight_t{0.8, 0.9, 0.8});
  EXPECT_NEAR(r, 7.1, tol);

  // Right, down, back:
  r = ripple::math::lerp(c, weight_t{0.6, 0.2, -0.3});
  EXPECT_NEAR(r, 4.52, tol);

  // Right, up, forward:
  r = ripple::math::lerp(c, weight_t{0.6, -0.7, 0.8});
  EXPECT_NEAR(r, 4.1, tol);  

  // Right, up, back:
  r = ripple::math::lerp(c, weight_t{0.3, -0.2, -0.3});
  EXPECT_NEAR(r, 5.12, tol);

  // left, down, forward:
  r = ripple::math::lerp(c, weight_t{-0.6, 0.75, 0.8});
  EXPECT_NEAR(r, 5.99, tol);

  // Left, down, back:
  r = ripple::math::lerp(c, weight_t{-0.4, 0.2, -0.3});
  EXPECT_NEAR(r, 4.92, tol);

  // Left, up, forward:
  r = ripple::math::lerp(c, weight_t{-0.1, -0.2, 0.2});
  EXPECT_NEAR(r, 5.42, tol);  

  // Left, up, back:
  r = ripple::math::lerp(c, weight_t{-0.6, -0.8, -0.9});
  EXPECT_NEAR(r, 2.6, tol);
}

// Tests that the interpolation works when the data type is stridable, but when
// the type owns it's data.
TEST(math_tests, can_lerp_3d_stridable_owned_data) {
  using namespace ripple;
  using data_t   = double;
  using type_t   = Vector<data_t, 2, contiguous_owned_t>;
  using c_t      = Vector<data_t, 2, contiguous_owned_t>;
  using weight_t = ripple::vec_3d_t<data_t>;
  host_block_3d_t<type_t> b(3, 3, 3);

  // First plane:
  *b(0, 0, 0) = c_t{1, 1}; *b(1, 0, 0) = c_t{2, 2}; *b(2, 0, 0) = c_t{3, 3};
  *b(0, 1, 0) = c_t{4, 4}; *b(1, 1, 0) = c_t{5, 5}; *b(2, 1, 0) = c_t{6, 6};
  *b(0, 2, 0) = c_t{7, 7}; *b(1, 2, 0) = c_t{8, 8}; *b(2, 2, 0) = c_t{9, 9};
  // Second plane:
  *b(0, 0, 1) = c_t{9, 9}; *b(1, 0, 1) = c_t{8, 8}; *b(2, 0, 1) = c_t{7, 7};
  *b(0, 1, 1) = c_t{6, 6}; *b(1, 1, 1) = c_t{5, 5}; *b(2, 1, 1) = c_t{4, 4};
  *b(0, 2, 1) = c_t{3, 3}; *b(1, 2, 1) = c_t{2, 2}; *b(2, 2, 1) = c_t{1, 1};
  // Third plane:
  *b(0, 0, 2) = c_t{1, 1}; *b(1, 0, 2) = c_t{2, 2}; *b(2, 0, 2) = c_t{3, 3};
  *b(0, 1, 2) = c_t{4, 4}; *b(1, 1, 2) = c_t{5, 5}; *b(2, 1, 2) = c_t{6, 6};
  *b(0, 2, 2) = c_t{7, 7}; *b(1, 2, 2) = c_t{8, 8}; *b(2, 2, 2) = c_t{9, 9};

  auto c = b(1, 1, 1);

  // Test simple case:
  auto r = ripple::math::lerp(c, weight_t{0.5, 0.5, 0.5});
  EXPECT_EQ(r[0], 5); EXPECT_EQ(r[1], 5);

  // Right, down, forward:
  r = ripple::math::lerp(c, weight_t{0.8, 0.9, 0.8});
  EXPECT_NEAR(r[0], 7.1, tol); EXPECT_NEAR(r[1], 7.1, tol);

  // Right, down, back:
  r = ripple::math::lerp(c, weight_t{0.6, 0.2, -0.3});
  EXPECT_NEAR(r[0], 4.52, tol); EXPECT_NEAR(r[1], 4.52, tol);

  // Right, up, forward:
  r = ripple::math::lerp(c, weight_t{0.6, -0.7, 0.8});
  EXPECT_NEAR(r[0], 4.1, tol);  EXPECT_NEAR(r[1], 4.1, tol);

  // Right, up, back:
  r = ripple::math::lerp(c, weight_t{0.3, -0.2, -0.3});
  EXPECT_NEAR(r[0], 5.12, tol); EXPECT_NEAR(r[1], 5.12, tol);

  // left, down, forward:
  r = ripple::math::lerp(c, weight_t{-0.6, 0.75, 0.8});
  EXPECT_NEAR(r[0], 5.99, tol); EXPECT_NEAR(r[1], 5.99, tol);

  // Left, down, back:
  r = ripple::math::lerp(c, weight_t{-0.4, 0.2, -0.3});
  EXPECT_NEAR(r[0], 4.92, tol);EXPECT_NEAR(r[1], 4.92, tol);

  // Left, up, forward:
  r = ripple::math::lerp(c, weight_t{-0.1, -0.2, 0.2});
  EXPECT_NEAR(r[0], 5.42, tol); EXPECT_NEAR(r[1], 5.42, tol);  

  // Left, up, back:
  r = ripple::math::lerp(c, weight_t{-0.6, -0.8, -0.9});
  EXPECT_NEAR(r[0], 2.6, tol); EXPECT_NEAR(r[1], 2.6, tol);
}

// Tests that the interpolation works when the data type is stridable, when
// the type does not own it's data, but it is contiguous.
TEST(math_tests, can_lerp_3d_stridable_contig_view_data) {
  using namespace ripple;
  using data_t   = double;
  using type_t   = Vector<data_t, 2, contiguous_view_t>;
  using c_t      = Vector<data_t, 2, contiguous_owned_t>;
  using weight_t = ripple::vec_3d_t<data_t>;
  host_block_3d_t<type_t> b(3, 3, 3);

  // First plane:
  *b(0, 0, 0) = c_t{1, 1}; *b(1, 0, 0) = c_t{2, 2}; *b(2, 0, 0) = c_t{3, 3};
  *b(0, 1, 0) = c_t{4, 4}; *b(1, 1, 0) = c_t{5, 5}; *b(2, 1, 0) = c_t{6, 6};
  *b(0, 2, 0) = c_t{7, 7}; *b(1, 2, 0) = c_t{8, 8}; *b(2, 2, 0) = c_t{9, 9};
  // Second plane:
  *b(0, 0, 1) = c_t{9, 9}; *b(1, 0, 1) = c_t{8, 8}; *b(2, 0, 1) = c_t{7, 7};
  *b(0, 1, 1) = c_t{6, 6}; *b(1, 1, 1) = c_t{5, 5}; *b(2, 1, 1) = c_t{4, 4};
  *b(0, 2, 1) = c_t{3, 3}; *b(1, 2, 1) = c_t{2, 2}; *b(2, 2, 1) = c_t{1, 1};
  // Third plane:
  *b(0, 0, 2) = c_t{1, 1}; *b(1, 0, 2) = c_t{2, 2}; *b(2, 0, 2) = c_t{3, 3};
  *b(0, 1, 2) = c_t{4, 4}; *b(1, 1, 2) = c_t{5, 5}; *b(2, 1, 2) = c_t{6, 6};
  *b(0, 2, 2) = c_t{7, 7}; *b(1, 2, 2) = c_t{8, 8}; *b(2, 2, 2) = c_t{9, 9};

  auto c = b(1, 1, 1);

  // Test simple case:
  auto r = ripple::math::lerp(c, weight_t{0.5, 0.5, 0.5});
  EXPECT_EQ(r[0], 5); EXPECT_EQ(r[1], 5);

  // Right, down, forward:
  r = ripple::math::lerp(c, weight_t{0.8, 0.9, 0.8});
  EXPECT_NEAR(r[0], 7.1, tol); EXPECT_NEAR(r[1], 7.1, tol);

  // Right, down, back:
  r = ripple::math::lerp(c, weight_t{0.6, 0.2, -0.3});
  EXPECT_NEAR(r[0], 4.52, tol); EXPECT_NEAR(r[1], 4.52, tol);

  // Right, up, forward:
  r = ripple::math::lerp(c, weight_t{0.6, -0.7, 0.8});
  EXPECT_NEAR(r[0], 4.1, tol);  EXPECT_NEAR(r[1], 4.1, tol);

  // Right, up, back:
  r = ripple::math::lerp(c, weight_t{0.3, -0.2, -0.3});
  EXPECT_NEAR(r[0], 5.12, tol); EXPECT_NEAR(r[1], 5.12, tol);

  // left, down, forward:
  r = ripple::math::lerp(c, weight_t{-0.6, 0.75, 0.8});
  EXPECT_NEAR(r[0], 5.99, tol); EXPECT_NEAR(r[1], 5.99, tol);

  // Left, down, back:
  r = ripple::math::lerp(c, weight_t{-0.4, 0.2, -0.3});
  EXPECT_NEAR(r[0], 4.92, tol);EXPECT_NEAR(r[1], 4.92, tol);

  // Left, up, forward:
  r = ripple::math::lerp(c, weight_t{-0.1, -0.2, 0.2});
  EXPECT_NEAR(r[0], 5.42, tol); EXPECT_NEAR(r[1], 5.42, tol);  

  // Left, up, back:
  r = ripple::math::lerp(c, weight_t{-0.6, -0.8, -0.9});
  EXPECT_NEAR(r[0], 2.6, tol); EXPECT_NEAR(r[1], 2.6, tol);
}

// Tests that the interpolation works when the data type is stridable, when
// the type does not own it's data, and it is strided.
TEST(math_tests, can_lerp_3d_stridable_strided_view_data) {
  using namespace ripple;
  using data_t   = double;
  using type_t   = Vector<data_t, 2, strided_view_t>;
  using c_t      = Vector<data_t, 2, contiguous_owned_t>;
  using weight_t = ripple::vec_3d_t<data_t>;
  host_block_3d_t<type_t> b(3, 3, 3);

  // First plane:
  *b(0, 0, 0) = c_t{1, 1}; *b(1, 0, 0) = c_t{2, 2}; *b(2, 0, 0) = c_t{3, 3};
  *b(0, 1, 0) = c_t{4, 4}; *b(1, 1, 0) = c_t{5, 5}; *b(2, 1, 0) = c_t{6, 6};
  *b(0, 2, 0) = c_t{7, 7}; *b(1, 2, 0) = c_t{8, 8}; *b(2, 2, 0) = c_t{9, 9};
  // Second plane:
  *b(0, 0, 1) = c_t{9, 9}; *b(1, 0, 1) = c_t{8, 8}; *b(2, 0, 1) = c_t{7, 7};
  *b(0, 1, 1) = c_t{6, 6}; *b(1, 1, 1) = c_t{5, 5}; *b(2, 1, 1) = c_t{4, 4};
  *b(0, 2, 1) = c_t{3, 3}; *b(1, 2, 1) = c_t{2, 2}; *b(2, 2, 1) = c_t{1, 1};
  // Third plane:
  *b(0, 0, 2) = c_t{1, 1}; *b(1, 0, 2) = c_t{2, 2}; *b(2, 0, 2) = c_t{3, 3};
  *b(0, 1, 2) = c_t{4, 4}; *b(1, 1, 2) = c_t{5, 5}; *b(2, 1, 2) = c_t{6, 6};
  *b(0, 2, 2) = c_t{7, 7}; *b(1, 2, 2) = c_t{8, 8}; *b(2, 2, 2) = c_t{9, 9};

  auto c = b(1, 1, 1);

  // Test simple case:
  auto r = ripple::math::lerp(c, weight_t{0.5, 0.5, 0.5});
  EXPECT_EQ(r[0], 5); EXPECT_EQ(r[1], 5);

  // Right, down, forward:
  r = ripple::math::lerp(c, weight_t{0.8, 0.9, 0.8});
  EXPECT_NEAR(r[0], 7.1, tol); EXPECT_NEAR(r[1], 7.1, tol);

  // Right, down, back:
  r = ripple::math::lerp(c, weight_t{0.6, 0.2, -0.3});
  EXPECT_NEAR(r[0], 4.52, tol); EXPECT_NEAR(r[1], 4.52, tol);

  // Right, up, forward:
  r = ripple::math::lerp(c, weight_t{0.6, -0.7, 0.8});
  EXPECT_NEAR(r[0], 4.1, tol);  EXPECT_NEAR(r[1], 4.1, tol);

  // Right, up, back:
  r = ripple::math::lerp(c, weight_t{0.3, -0.2, -0.3});
  EXPECT_NEAR(r[0], 5.12, tol); EXPECT_NEAR(r[1], 5.12, tol);

  // left, down, forward:
  r = ripple::math::lerp(c, weight_t{-0.6, 0.75, 0.8});
  EXPECT_NEAR(r[0], 5.99, tol); EXPECT_NEAR(r[1], 5.99, tol);

  // Left, down, back:
  r = ripple::math::lerp(c, weight_t{-0.4, 0.2, -0.3});
  EXPECT_NEAR(r[0], 4.92, tol);EXPECT_NEAR(r[1], 4.92, tol);

  // Left, up, forward:
  r = ripple::math::lerp(c, weight_t{-0.1, -0.2, 0.2});
  EXPECT_NEAR(r[0], 5.42, tol); EXPECT_NEAR(r[1], 5.42, tol);  

  // Left, up, back:
  r = ripple::math::lerp(c, weight_t{-0.6, -0.8, -0.9});
  EXPECT_NEAR(r[0], 2.6, tol); EXPECT_NEAR(r[1], 2.6, tol);
}


#endif // RIPPLE_TESTS_MATH_LERP_TESTS_HPP