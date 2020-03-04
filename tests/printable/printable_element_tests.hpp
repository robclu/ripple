//==--- ripple/tests/printable/printable_element_tests.hpp - -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  printable_element_tests.hpp
/// \brief This file defines tests for printable elements.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_PRINTABLE_PRINTABLE_ELEMENT_TESTS_HPP
#define RIPPLE_TESTS_PRINTABLE_PRINTABLE_ELEMENT_TESTS_HPP

#include <ripple/viz/printable/printable_element.hpp>
#include <gtest/gtest.h>

TEST(printable_printable_element_tests, can_create_scalar_element) {
  ripple::viz::PrintableElement p(
    "test", ripple::viz::PrintableElement::AttributeKind::scalar, 4.4
  );

  auto& v = p.values().front();
  EXPECT_EQ(p.name(), "test");
  EXPECT_EQ(p.kind(), ripple::viz::PrintableElement::AttributeKind::scalar);
  EXPECT_EQ(v       , 4.4);
}

TEST(printable_printable_element_tests, can_create_vector_element) {
  ripple::viz::PrintableElement p(
    "test", ripple::viz::PrintableElement::AttributeKind::vector, 3.14, 0.2, 0.7
  );

  auto& v = p.values();
  EXPECT_EQ(p.name(), "test");
  EXPECT_EQ(p.kind(), ripple::viz::PrintableElement::AttributeKind::vector);
  EXPECT_EQ(v[0]    , 3.14);
  EXPECT_EQ(v[1]    , 0.2);
  EXPECT_EQ(v[2]    , 0.7);
}

TEST(printable_printable_element_tests, adds_default_zero_vectors) {
  ripple::viz::PrintableElement p(
    "test", ripple::viz::PrintableElement::AttributeKind::vector, 3.14
  );

  auto& v = p.values();
  EXPECT_EQ(p.name(), "test");
  EXPECT_EQ(p.kind(), ripple::viz::PrintableElement::AttributeKind::vector);
  EXPECT_EQ(v[0]    , 3.14);
  EXPECT_EQ(v[1]    , 0.0);
  EXPECT_EQ(v[2]    , 0.0);
}

TEST(printable_printable_element_tests, can_compare_elements) {
  ripple::viz::PrintableElement p1(
    "test", ripple::viz::PrintableElement::AttributeKind::vector, 3.14
  );
  ripple::viz::PrintableElement p2(
    "test", ripple::viz::PrintableElement::AttributeKind::vector, 3.14
  );

  EXPECT_TRUE(p1 == p2);
}

TEST(printable_printable_element_tests, can_compare_not_found_elements) {
  auto p1 = ripple::viz::PrintableElement::not_found();
  auto p2 = ripple::viz::PrintableElement::not_found();

  EXPECT_TRUE(p1 == p2);

  ripple::viz::PrintableElement p3(
    "test", ripple::viz::PrintableElement::AttributeKind::vector, 3.14
  );

  EXPECT_TRUE(p1 != p3);
  EXPECT_TRUE(p2 != p3);
}

#endif // RIPPLE_TESTS_PRINTABLE_PRINTABLE_ELEMENT_TESTS_HPP
