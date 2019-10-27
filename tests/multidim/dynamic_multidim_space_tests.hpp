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

TEST(multidim_dynamic_multidim_space, size_and_step_1d) {
  using space_t = ripple::DynamicMultidimSpace<1>;

  auto space = space_t{10};
  EXPECT_EQ(space.size()             , size_t{10});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );

  // Test that for dim 0 SOA makes no difference:
  EXPECT_EQ(space.step(ripple::dim_x, 3), std::size_t{1});

  space[0] = 30;
  EXPECT_EQ(space.size()                , size_t{30});
  EXPECT_EQ(space.size(ripple::dim_x)   , size_t{30});
  EXPECT_EQ(space.step(ripple::dim_x)   , size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_x, 3), size_t{1} );
}

TEST(multidim_dynamic_multidim_space, size_and_step_2d) {
  using space_t = ripple::DynamicMultidimSpace<2>;

  auto space = space_t{10, 20};
  EXPECT_EQ(space.size()             , size_t{200});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{10} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{20} );
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{10} );

  // Test step sizes for SOA:
  EXPECT_EQ(space.step(ripple::dim_x, 3), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y, 3), size_t{30});
  EXPECT_EQ(space.step(ripple::dim_x, 4), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y, 4), size_t{40});

  space[0] = 5;
  space[1] = 10;
  EXPECT_EQ(space.size()             , size_t{50});
  EXPECT_EQ(space.size(ripple::dim_x), size_t{5} );
  EXPECT_EQ(space.size(ripple::dim_y), size_t{10});
  EXPECT_EQ(space.step(ripple::dim_x), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y), size_t{5} );

  // Test step sizes for SOA:
  EXPECT_EQ(space.step(ripple::dim_x, 2), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y, 2), size_t{10});
  EXPECT_EQ(space.step(ripple::dim_x, 5), size_t{1} );
  EXPECT_EQ(space.step(ripple::dim_y, 5), size_t{25});
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

  // Test step sizes for SOA:
  EXPECT_EQ(space.step(ripple::dim_x, 3), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y, 3), size_t{30} );
  EXPECT_EQ(space.step(ripple::dim_z, 3), size_t{600});
  EXPECT_EQ(space.step(ripple::dim_x, 4), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y, 4), size_t{40} );
  EXPECT_EQ(space.step(ripple::dim_z, 4), size_t{800});

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

  // Test step sizes for SOA:
  EXPECT_EQ(space.step(ripple::dim_x, 2), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y, 2), size_t{10} );
  EXPECT_EQ(space.step(ripple::dim_z, 2), size_t{100});
  EXPECT_EQ(space.step(ripple::dim_x, 7), size_t{1}  );
  EXPECT_EQ(space.step(ripple::dim_y, 7), size_t{35} );
  EXPECT_EQ(space.step(ripple::dim_z, 7), size_t{350});
}

TEST(multidim_dynamic_multidim_space, padding_always_zero) {
  using space_t = ripple::DynamicMultidimSpace<3>;
  auto space    = space_t{10, 20, 5};

  EXPECT_EQ(space.padding(ripple::dim_x), std::size_t{0});
  EXPECT_EQ(space.padding(ripple::dim_y), std::size_t{0});
  EXPECT_EQ(space.padding(ripple::dim_z), std::size_t{0});
}

#endif // RIPPLE_TESTS_MULTIDIM_DYNAMIC_MULTIDIM_SPACE_TESTS_HPP

