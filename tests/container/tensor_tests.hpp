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

TEST(container_tensor, can_create_tensor) {
  ripple::host_tensor_1d_t<float> t(20);
  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20));
}

#endif // RIPPLE_TESTS_CONTAINER_TENSOR_TESTS_HPP

