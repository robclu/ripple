//==--- ripple/tests/container/tensor_tests.hpp ------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_tests.hpp
/// \brief This file defines tests for tensor functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_TENSOR_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_TENSOR_TESTS_HPP

#include <ripple/container/host_tensor.hpp>
#include <gtest/gtest.h>

//==--- [creation] ---------------------------------------------------------==//

TEST(container_tensor, can_create_tensor_1d) {
  ripple::host_tensor_1d_t<float> t(20);
  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20));
}

TEST(container_tensor, can_create_tensor_2d) {
  ripple::host_tensor_2d_t<int> t(20, 20);
  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20 * 20));
}

TEST(container_tensor, can_create_tensor_3d) {
  ripple::host_tensor_1d_t<double> t(20, 20, 20);
  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20 * 20 * 20));
}

//==-- [access simple types] -----------------------------------------------==//

TEST(container_tensor, can_access_simple_elements_1d) {
  ripple::host_tensor_1d_t<float> t(20);
  for (auto i : ripple::range(t.size())) {
    t(i) = static_cast<float>(i);
  }
  for (auto i : ripple::range(t.size())) {
    EXPECT_EQ(t(i), static_cast<float>(i));
  }
}

TEST(container_tensor, can_access_simple_elements_2d) {
  ripple::host_tensor_2d_t<float> t(20, 20);
  for (auto j : ripple::range(t.size(ripple::dim_y))) {
    for (auto i : ripple::range(t.size(ripple::dim_x))) {
      t(i, j) = static_cast<float>(i);
    }
  }
  for (auto j : ripple::range(t.size(ripple::dim_y))) {
    for (auto i : ripple::range(t.size(ripple::dim_x))) {
      EXPECT_EQ(t(i, j), static_cast<float>(i));
    }
  }
}

TEST(container_tensor, can_access_simple_elements_3d) {
  ripple::host_tensor_3d_t<float> t(20, 20, 20);
  for (auto k : ripple::range(t.size(ripple::dim_z))) {
    for (auto j : ripple::range(t.size(ripple::dim_y))) {
      for (auto i : ripple::range(t.size(ripple::dim_x))) {
        t(i, j, k) = static_cast<float>(i);
      }
    }
  }
  for (auto k : ripple::range(t.size(ripple::dim_z))) {
    for (auto j : ripple::range(t.size(ripple::dim_y))) {
      for (auto i : ripple::range(t.size(ripple::dim_x))) {
        EXPECT_EQ(t(i, j, k), static_cast<float>(i));
      }
    }
  }
}

//==--- [access vec types] -------------------------------------------------==//

TEST(container_tensor, can_access_vec_elements_1d) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_tensor_1d_t<vec_t> t(20);

  for (auto i : ripple:range(t.size())) {
    auto& v = t(i);
    for (auto element : range(v.size())) {
      v[element] = static_cast<float>(element) * i;
    }
  }

  for (auto i : ripple::range(t.size())) {
    auto& v_ref     = t(i);
    auto  v_non_ref = t(i);
    
    EXEPECT_EQ(v.size(), v_non_ref.size());

    for (auto element : ripple::range(v.size())) {
      EXPECT_EQ(v_non_ref[element], static_cast<float>(element) * i);
      EXPECT_EQ(v_ref[element]    , static_cast<float>(element) * i);
    }
  }
}

TEST(container_tensor, can_access_vec_elements_2d) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_tensor_2d_t<vec_t> t(20, 20);

  for (auto j : ripple::range(t.size(ripple::dim_y))) {
    for (auto i : ripple::range(t.size(ripple::dim_x))) {
      auto& v = t(i, j);
      for (auto element : range(v.size())) {
        v[element] = static_cast<float>(element) * i;
      }
    }
  }


  for (auto j : ripple::range(t.size(ripple::dim_y))) {
    for (auto i : ripple::range(t.size(ripple::dim_x))) {
      auto& v_ref     = t(i, j);
      auto  v_non_ref = t(i, j);
    
      EXEPECT_EQ(v.size(), v_non_ref.size());

      for (auto element : ripple::range(v.size())) {
        EXPECT_EQ(v_non_ref[element], static_cast<float>(element) * i);
        EXPECT_EQ(v_ref[element]    , static_cast<float>(element) * i);
      }
    }
  }
}

TEST(container_tensor, can_access_vec_elements_3d) {
  using vec_t = ripple::Vec<float, 3>;
  ripple::host_tensor_3d_t<vec_t> t(20, 20, 10);

  for (auto k : ripple::range(t.size(ripple::dim_z))) {
    for (auto j : ripple::range(t.size(ripple::dim_y))) {
      for (auto i : ripple::range(t.size(ripple::dim_x))) {
        auto& v = t(i, j, k);
        for (auto element : range(v.size())) {
          v[element] = static_cast<float>(element) * j;
        }
      }
    }
  }

  for (auto k : ripple::range(t.size(ripple::dim_z))) {
    for (auto j : ripple::range(t.size(ripple::dim_y))) {
      for (auto i : ripple::range(t.size(ripple::dim_x))) {
        auto& v_ref     = t(i, j, k);
        auto  v_non_ref = t(i, j, k);
    
        EXEPECT_EQ(v.size(), v_non_ref.size());

        for (auto element : ripple::range(v.size())) {
          EXPECT_EQ(v_non_ref[element], static_cast<float>(element) * j);
          EXPECT_EQ(v_ref[element]    , static_cast<float>(element) * j);
        }
      }
    }
  }
}


//==--- [access soa vec types] ---------------------------------------------==//

TEST(container_tensor, can_access_soa_vec_elements_1d) {
  using vec_t = ripple::SoaVec<float, 3>;
  ripple::host_tensor_1d_t<vec_t> t(20);

  for (auto i : ripple:range(t.size())) {
    auto& v = t(i);
    for (auto element : range(v.size())) {
      v[element] = static_cast<float>(element) * i;
    }
  }

  for (auto i : ripple::range(t.size())) {
    auto& v_ref     = t(i);
    auto  v_non_ref = t(i);
    
    EXEPECT_EQ(v.size(), v_non_ref.size());

    for (auto element : ripple::range(v.size())) {
      EXPECT_EQ(v_non_ref[element], static_cast<float>(element) * i);
      EXPECT_EQ(v_ref[element]    , static_cast<float>(element) * i);
    }
  }
}


TEST(container_tensor, can_access_soa_vec_elements_2d) {
  using vec_t = ripple::SoaVec<float, 3>;
  ripple::host_tensor_2d_t<vec_t> t(20, 20);

  for (auto j : ripple::range(t.size(ripple::dim_y))) {
    for (auto i : ripple::range(t.size(ripple::dim_x))) {
      auto& v = t(i, j);
      for (auto element : range(v.size())) {
        v[element] = static_cast<float>(element) * i;
      }
    }
  }

  for (auto j : ripple::range(t.size(ripple::dim_y))) {
    for (auto i : ripple::range(t.size(ripple::dim_x))) {
      auto& v_ref     = t(i, j);
      auto  v_non_ref = t(i, j);
    
      EXEPECT_EQ(v.size(), v_non_ref.size());

      for (auto element : ripple::range(v.size())) {
        EXPECT_EQ(v_non_ref[element], static_cast<float>(element) * i);
        EXPECT_EQ(v_ref[element]    , static_cast<float>(element) * i);
      }
    }
  }
}

TEST(container_tensor, can_access_soa_vec_elements_3d) {
  using vec_t = ripple::SoaVec<float, 3>;
  ripple::host_tensor_3d_t<vec_t> t(20, 20, 10);

  for (auto k : ripple::range(t.size(ripple::dim_z))) {
    for (auto j : ripple::range(t.size(ripple::dim_y))) {
      for (auto i : ripple::range(t.size(ripple::dim_x))) {
        auto& v = t(i, j, k);
        for (auto element : range(v.size())) {
          v[element] = static_cast<float>(element) * j;
        }
      }
    }
  }

  for (auto k : ripple::range(t.size(ripple::dim_z))) {
    for (auto j : ripple::range(t.size(ripple::dim_y))) {
      for (auto i : ripple::range(t.size(ripple::dim_x))) {
        auto& v_ref     = t(i, j, k);
        auto  v_non_ref = t(i, j, k);
    
        EXEPECT_EQ(v.size(), v_non_ref.size());

        for (auto element : ripple::range(v.size())) {
          EXPECT_EQ(v_non_ref[element], static_cast<float>(element) * j);
          EXPECT_EQ(v_ref[element]    , static_cast<float>(element) * j);
        }
      }
    }
  }
}


#endif // RIPPLE_TESTS_CONTAINER_TENSOR_TESTS_HPP

