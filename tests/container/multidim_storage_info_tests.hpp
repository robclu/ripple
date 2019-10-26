//==--- ../tests/container/multidim_storage_info_tests.cpp - -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multidim_storage_info_tests.hpp
/// \brief This file contains tests for multidimensional storage information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_MULTIDIM_STORAGE_INFO_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_MULTIDIM_STORAGE_INFO_TESTS_HPP

#include <ripple/utility/dim.hpp>
#include <ripple/container/multidim_storage_info.hpp>
#include <gtest/gtest.h>

TEST(container_multidim_storage_info, size_and_step_1d) {
  using storage_t = ripple::MultidimStorageInfo<1>;

  auto storage = storage_t{10};
  EXPECT_EQ(storage.size()             , size_t{10});
  EXPECT_EQ(storage.size(ripple::dim_x), size_t{10});
  EXPECT_EQ(storage.step(ripple::dim_x), size_t{1} );

  // Test that for dim 0 SOA makes no difference:
  EXPECT_EQ(storage.step(ripple::dim_x, 3), std::size_t{1});

  storage[0] = 30;
  EXPECT_EQ(storage.size()                , size_t{30});
  EXPECT_EQ(storage.size(ripple::dim_x)   , size_t{30});
  EXPECT_EQ(storage.step(ripple::dim_x)   , size_t{1} );
  EXPECT_EQ(storage.step(ripple::dim_x, 3), size_t{1} );
}

TEST(container_multidim_storage_info, size_and_step_2d) {
  using storage_t = ripple::MultidimStorageInfo<2>;

  auto storage = storage_t{10, 20};
  EXPECT_EQ(storage.size()             , size_t{200});
  EXPECT_EQ(storage.size(ripple::dim_x), size_t{10} );
  EXPECT_EQ(storage.size(ripple::dim_y), size_t{20} );
  EXPECT_EQ(storage.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(storage.step(ripple::dim_y), size_t{10} );

  // Test step sizes for SOA:
  EXPECT_EQ(storage.step(ripple::dim_x, 3), size_t{1} );
  EXPECT_EQ(storage.step(ripple::dim_y, 3), size_t{30});
  EXPECT_EQ(storage.step(ripple::dim_x, 4), size_t{1} );
  EXPECT_EQ(storage.step(ripple::dim_y, 4), size_t{40});


  storage[0] = 5;
  storage[1] = 10;
  EXPECT_EQ(storage.size()             , size_t{50});
  EXPECT_EQ(storage.size(ripple::dim_x), size_t{5} );
  EXPECT_EQ(storage.size(ripple::dim_y), size_t{10});
  EXPECT_EQ(storage.step(ripple::dim_x), size_t{1} );
  EXPECT_EQ(storage.step(ripple::dim_y), size_t{5} );

  // Test step sizes for SOA:
  EXPECT_EQ(storage.step(ripple::dim_x, 2), size_t{1} );
  EXPECT_EQ(storage.step(ripple::dim_y, 2), size_t{10});
  EXPECT_EQ(storage.step(ripple::dim_x, 5), size_t{1} );
  EXPECT_EQ(storage.step(ripple::dim_y, 5), size_t{25});
}

TEST(container_multidim_storage_info, size_and_step_3d) {
  using storage_t = ripple::MultidimStorageInfo<3>;

  auto storage = storage_t{10, 20, 5};
  EXPECT_EQ(storage.size()             , size_t{1000});
  EXPECT_EQ(storage.size(ripple::dim_x), size_t{10}  );
  EXPECT_EQ(storage.size(ripple::dim_y), size_t{20}  );
  EXPECT_EQ(storage.size(ripple::dim_z), size_t{5}   );
  EXPECT_EQ(storage.step(ripple::dim_x), size_t{1}   );
  EXPECT_EQ(storage.step(ripple::dim_y), size_t{10}  );
  EXPECT_EQ(storage.step(ripple::dim_z), size_t{200} );

  // Test step sizes for SOA:
  EXPECT_EQ(storage.step(ripple::dim_x, 3), size_t{1}  );
  EXPECT_EQ(storage.step(ripple::dim_y, 3), size_t{30} );
  EXPECT_EQ(storage.step(ripple::dim_z, 3), size_t{600});
  EXPECT_EQ(storage.step(ripple::dim_x, 4), size_t{1}  );
  EXPECT_EQ(storage.step(ripple::dim_y, 4), size_t{40} );
  EXPECT_EQ(storage.step(ripple::dim_z, 4), size_t{800});

  storage[0] = 5;
  storage[1] = 10;
  storage[2] = 2;
  EXPECT_EQ(storage.size()             , size_t{100});
  EXPECT_EQ(storage.size(ripple::dim_x), size_t{5}  );
  EXPECT_EQ(storage.size(ripple::dim_y), size_t{10} );
  EXPECT_EQ(storage.size(ripple::dim_z), size_t{2}  );
  EXPECT_EQ(storage.step(ripple::dim_x), size_t{1}  );
  EXPECT_EQ(storage.step(ripple::dim_y), size_t{5}  );
  EXPECT_EQ(storage.step(ripple::dim_z), size_t{50} );

  // Test step sizes for SOA:
  EXPECT_EQ(storage.step(ripple::dim_x, 2), size_t{1}  );
  EXPECT_EQ(storage.step(ripple::dim_y, 2), size_t{10} );
  EXPECT_EQ(storage.step(ripple::dim_z, 2), size_t{100});
  EXPECT_EQ(storage.step(ripple::dim_x, 7), size_t{1}  );
  EXPECT_EQ(storage.step(ripple::dim_y, 7), size_t{35} );
  EXPECT_EQ(storage.step(ripple::dim_z, 7), size_t{350});
}

#endif // RIPPLE_TESTS_CONTAINER_MULTIDIM_STORAGE_INFO_TESTS_HPP

