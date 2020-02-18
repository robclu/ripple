//==--- ripple/tests/state/state_element_tests.hpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_element_tests.hpp
/// \brief This file defines tests for state elements.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STATE_STATE_ELEMENT_TESTS_HPP
#define RIPPLE_TESTS_STATE_STATE_ELEMENT_TESTS_HPP

#include <ripple/fvm/state/state_element.hpp>
#include <gtest/gtest.h>

using namespace ripple::fv::state;

TEST(state_state_element, can_create_density_element) {
  const auto rho = 4.4_rho;
  EXPECT_EQ(rho.value, 4.4);
  EXPECT_EQ(std::string(rho.name), std::string("rho"));

  const auto mrho = -3.7_rho;
  EXPECT_EQ(mrho.value, -3.7);
  EXPECT_EQ(std::string(mrho.name), std::string("rho"));
}

TEST(state_state_element, can_create_pressure_element) {
  const auto p = 4.4_p;
  EXPECT_EQ(p.value, 4.4);
  EXPECT_EQ(std::string(p.name), std::string("pressure"));

  const auto mp = -3.7_p;
  EXPECT_EQ(mp.value, -3.7);
  EXPECT_EQ(std::string(mp.name), std::string("pressure"));
}

TEST(state_state_element, can_create_v_x_element) {
  const auto vx = 4.4_v_x;
  EXPECT_EQ(vx.value, 4.4);
  EXPECT_EQ(std::string(vx.name), std::string("v_x"));

  const auto mvx = -3.7_v_x;
  EXPECT_EQ(mvx.value, -3.7);
  EXPECT_EQ(std::string(mvx.name), std::string("v_x"));
}

TEST(state_state_element, can_create_v_y_element) {
  const auto vy = 4.4_v_y;
  EXPECT_EQ(vy.value, 4.4);
  EXPECT_EQ(std::string(vy.name), std::string("v_y"));

  const auto mvy = -3.7_v_y;
  EXPECT_EQ(mvy.value, -3.7);
  EXPECT_EQ(std::string(mvy.name), std::string("v_y"));
}

TEST(state_state_element, can_create_v_z_element) {
  const auto vz = 4.4_v_z;
  EXPECT_EQ(vz.value, 4.4);
  EXPECT_EQ(std::string(vz.name), std::string("v_z"));

  const auto mvz = -3.7_v_z;
  EXPECT_EQ(mvz.value, -3.7);
  EXPECT_EQ(std::string(mvz.name), std::string("v_z"));
}

#endif // RIPPLE_TESTS_STATE_STATE_ELEMENT_TESTS_HPP

