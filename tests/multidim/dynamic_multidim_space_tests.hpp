//==--- ../multidim/dynamic_multidim_space_tests.cpp ------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dynamic_multidim_space_tests.hpp
/// \brief This file contains tests for a dynamic multidimensional space.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_MULTIDIM_DYNAMIC_MULTIDIM_SPACE_TESTS_HPP
#define RIPPLE_TESTS_MULTIDIM_DYNAMIC_MULTIDIM_SPACE_TESTS_HPP

#include <ripple/multidim/dynamic_multidim_space.hpp>
#include <ripple/utility/dim.hpp>
#include <gtest/gtest.h>

//==--- [no padding] -------------------------------------------------------==//

TEST(multidim_dynamic_multidim_space, size_and_step_1d) {
  using space_t = ripple::DynamicMultidimSpace<1>;

  auto space = space_t{10};
  EXPECT_EQ(space.size()             , size_t{10});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );

  space[0] = 30;
  EXPECT_EQ(space.size()             , size_t{30});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{30});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );
}

TEST(multidim_dynamic_multidim_space, size_and_step_2d) {
  using space_t = ripple::DynamicMultidimSpace<2>;

  auto space = space_t{10, 20};
  EXPECT_EQ(space.size()             , size_t{200});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20} );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10} );

  space[0] = 5;
  space[1] = 10;
  EXPECT_EQ(space.size()             , size_t{50});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{5} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{10});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{5} );
}

TEST(multidim_dynamic_multidim_space, size_and_step_3d) {
  using space_t = ripple::DynamicMultidimSpace<3>;

  auto space = space_t{10, 20, 5};
  EXPECT_EQ(space.size()             , size_t{1000});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10}  );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20}  );
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5}   );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}   );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10}  );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{200} );

  space[0] = 5;
  space[1] = 10;
  space[2] = 2;
  EXPECT_EQ(space.size()             , size_t{100});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{5}  );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{10} );
  EXPECT_EQ(space.size(ripple::dim_z), size_t{2}  );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{5}  );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{50} );
}

TEST(multidim_dynamic_multidim_space, can_resize) {
  using space_t = ripple::DynamicMultidimSpace<3>;

  auto space = space_t{10, 20, 5};
  EXPECT_EQ(space.size()             , size_t{1000});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10}  );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20}  );
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5}   );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}   );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10}  );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{200} );

  space.resize(5, 10, 2);
  EXPECT_EQ(space.size()             , size_t{100});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{5}  );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{10} );
  EXPECT_EQ(space.size(ripple::dim_z), size_t{2}  );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{5}  );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{50} );

  space.resize(22, 3);
  EXPECT_EQ(space.size()             , size_t{132});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{22} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{3}  );
  EXPECT_EQ(space.size(ripple::dim_z), size_t{2}  );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{22} );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{66} );
}

TEST(multidim_dynamic_multidim_space, default_padding_is_zero) {
  using space_t = ripple::DynamicMultidimSpace<3>;
  auto space    = space_t{10, 20, 5};

  EXPECT_EQ(space.padding()          , size_t{0});
  EXPECT_EQ(space.size()             , size_t{10 * 20 * 5});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20});
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5} );
}

//==--- [padding] ----------------------------------------------------------==//

TEST(multidim_dynamic_multidim_space, size_and_step_with_padding_1d) {
  using space_t = ripple::DynamicMultidimSpace<1>;

  std::size_t pad     = 2;
  std::size_t dim_pad = pad * 2;
  auto space          = space_t{pad, 10};
  EXPECT_EQ(space.size()             , size_t{10 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10 + dim_pad});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});

  EXPECT_EQ(space.internal_size()             , std::size_t{10});
  EXPECT_EQ(space.internal_size(ripple::dim_x), std::size_t{10});

  space[0] = 30;
  EXPECT_EQ(space.size()             , size_t{30 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{30 + dim_pad});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});

  EXPECT_EQ(space.internal_size()             , std::size_t{30});
  EXPECT_EQ(space.internal_size(ripple::dim_x), std::size_t{30});
}

TEST(multidim_dynamic_multidim_space, size_and_step_with_padding_2d) {
  using space_t = ripple::DynamicMultidimSpace<2>;

  std::size_t pad     = 2;
  std::size_t pad_dim = pad * 2;
  auto space          = space_t{pad, 10, 20};
  EXPECT_EQ(space.size()             , size_t{(10 + pad_dim) * (20 + pad_dim)});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10 + pad_dim} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20 + pad_dim} );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10 + pad_dim} );

  EXPECT_EQ(space.internal_size()             , size_t{10 * 20});
  EXPECT_EQ(space.internal_size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.internal_size(ripple::dim_y), size_t{20});

  space[0] = 5;
  space[1] = 10;
  EXPECT_EQ(space.size()             , size_t{(5 + pad_dim) * (10 + pad_dim)});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{5  + pad_dim} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{10 + pad_dim});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{5 + pad_dim} );

  EXPECT_EQ(space.internal_size()             , size_t{5 * 10});
  EXPECT_EQ(space.internal_size(ripple::dim_x), size_t{5});
  EXPECT_EQ(space.internal_size(ripple::dim_y), size_t{10});
}

TEST(multidim_dynamic_multidim_space, size_and_step_with_padding_3d) {
  using space_t = ripple::DynamicMultidimSpace<3>;

  std::size_t pad     = 3;
  std::size_t dim_pad = pad * 2;
  auto space          = space_t{pad, 10, 20, 5};
  
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

  space[0] = 5;
  space[1] = 10;
  space[2] = 2;
  
  size = size_t{(5 + dim_pad) * (10 + dim_pad) * (2 + dim_pad)};
  EXPECT_EQ(space.size()             , size);
  EXPECT_EQ(space.size(ripple::dim_x), size_t{5 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_y), size_t{10 + dim_pad});
  EXPECT_EQ(space.size(ripple::dim_z), size_t{2 + dim_pad});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1});
  EXPECT_EQ(space.step(ripple::dim_y), size_t{5 + dim_pad}  );
  EXPECT_EQ(space.step(ripple::dim_z), size_t{(5 + dim_pad) * (10 + dim_pad)});

  EXPECT_EQ(space.internal_size()             , size_t{100});
  EXPECT_EQ(space.internal_size(ripple::dim_x), size_t{5});
  EXPECT_EQ(space.internal_size(ripple::dim_y), size_t{10});
  EXPECT_EQ(space.internal_size(ripple::dim_z), size_t{2});
}


TEST(multidim_dynamic_multidim_space, works_with_custom_padding) {
  using space_t         = ripple::DynamicMultidimSpace<3>;
  constexpr auto space  = space_t{3, 10, 20, 5};
  const auto size       = size_t{(10 + 6) * (20 + 6) * (5 + 6)};
  EXPECT_EQ(space.padding()          , size_t{3});
  EXPECT_EQ(space.size()             , size);
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10 + 6});
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20 + 6});
  EXPECT_EQ(space.size(ripple::dim_z), size_t{5 + 6});
}


#endif // RIPPLE_TESTS_MULTIDIM_DYNAMIC_MULTIDIM_SPACE_TESTS_HPP

