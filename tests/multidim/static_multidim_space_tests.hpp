//==--- ../multidim/static_multidim_space_tests.cpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  static_multidim_space_tests.hpp
/// \brief This file contains tests for a static multidimensional space.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_MULTIDIM_STATIC_MULTIDIM_SPACE_TESTS_HPP
#define RIPPLE_TESTS_MULTIDIM_STATIC_MULTIDIM_SPACE_TESTS_HPP

#include <ripple/multidim/static_multidim_space.hpp>
#include <ripple/utility/dim.hpp>
#include <gtest/gtest.h>

TEST(multidim_static_multidim_space, size_and_step_1d) {
  using space_t        = ripple::StaticMultidimSpace<10>;
  constexpr auto space = space_t();

  EXPECT_EQ(space.size()             , size_t{10});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );
}


TEST(multidim_static_multidim_space, size_and_step_2d) {
  using space_t        = ripple::StaticMultidimSpace<10, 20>;
  constexpr auto space = space_t();

  EXPECT_EQ(space.size()             , size_t{200});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20} );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10} );
}


TEST(multidim_static_multidim_space, size_and_step_3d) {
  using space_t        = ripple::StaticMultidimSpace<10, 20, 5>;
  constexpr auto space = space_t();

  EXPECT_EQ(space.size()             , size_t{1000});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10}  );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20}  );
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5}   );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}   );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10}  );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{200} );
}

TEST(multidim_static_multidim_space, default_padding_is_zero) {
  using space_t        = ripple::StaticMultidimSpace<10, 20, 5>;
  constexpr auto space = space_t();

  EXPECT_EQ(space.padding()          , size_t{0});
  EXPECT_EQ(space.size()             , size_t{10 * 20 * 5});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20});
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5} );
}

//==--- [padding] ----------------------------------------------------------==//

TEST(multidim_static_multidim_space, size_and_step_with_padding_1d) {
  using space_t = ripple::StaticMultidimSpace<10>;

  std::size_t pad     = 2;
  std::size_t dim_pad = pad * 2;
  const auto space    = space_t{pad};
  EXPECT_EQ(space.size()             , size_t{10 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10 + dim_pad});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});

  EXPECT_EQ(space.internal_size()             , std::size_t{10});
  EXPECT_EQ(space.internal_size(ripple::dim_x), std::size_t{10});
}

TEST(multidim_static_multidim_space, size_and_step_with_padding_2d) {
  using space_t = ripple::StaticMultidimSpace<10, 20>;

  constexpr std::size_t pad     = 2;
  constexpr std::size_t pad_dim = pad * 2;
  constexpr auto space          = space_t{pad};
  EXPECT_EQ(space.size()             , size_t{(10 + pad_dim) * (20 + pad_dim)});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10 + pad_dim} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20 + pad_dim} );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10 + pad_dim} );

  EXPECT_EQ(space.internal_size()             , size_t{10 * 20});
  EXPECT_EQ(space.internal_size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.internal_size(ripple::dim_y), size_t{20});
}

TEST(multidim_static_multidim_space, size_and_step_with_padding_3d) {
  using space_t = ripple::StaticMultidimSpace<10, 20, 5>;

  std::size_t pad     = 3;
  std::size_t dim_pad = pad * 2;
  const auto space    = space_t{pad};
  
  auto size = size_t{(10 + dim_pad) * (20 + dim_pad) * (5 + dim_pad)};
  EXPECT_EQ(space.size()             , size);
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5 + dim_pad});

  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10 + dim_pad});
  EXPECT_EQ(space.step(ripple::dim_z), size_t{(10 + dim_pad) * (20 + dim_pad)});

  EXPECT_EQ(space.internal_size()             , size_t{1000});
  EXPECT_EQ(space.internal_size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.internal_size(ripple::dim_y), size_t{20});
  EXPECT_EQ(space.internal_size(ripple::dim_z), size_t{5});
}

#endif // RIPPLE_TESTS_MULTIDIM_STATIC_MULTIDIM_SPACE_TESTS_HPP
