//==--- ripple/core/tests/storage/storage_traits_tests.hpp --------*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits_tests.hpp
/// \brief This file contains tests for storage traits.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP
#define RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP

#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/storage/stridable_layout.hpp>
#include <ripple/core/utility/number.hpp>
#include <gtest/gtest.h>

TEST(storage_storage_traits, can_determine_layout_types) {
  struct Test {};

  auto b1 = ripple::is_storage_layout_v<ripple::strided_view_t>;
  auto b2 = ripple::is_storage_layout_v<ripple::contiguous_view_t>;
  auto b3 = ripple::is_storage_layout_v<int>;
  auto b4 = ripple::is_storage_layout_v<Test>;
  auto b5 = ripple::is_storage_layout_v<ripple::contiguous_owned_t>;

  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
  EXPECT_FALSE(b4);
  EXPECT_TRUE(b5);
}

// Usual struct:
struct Normal {};
// Struct with multiple types.
template <typename... Ts>
struct Params {};
// Struct with a default type.
template <typename T, typename U = ripple::contiguous_view_t>
struct Default {};
// Struct with int.
template <int I, typename... Ts>
struct WithInt {};

TEST(storage_storage_traits, can_determine_if_type_has_storage_layout) {
  using mult_2_t   = Params<int, ripple::contiguous_view_t>;
  using mult_3_t   = Params<ripple::strided_view_t, int, float>;
  using default_t  = Default<int>;
  using with_int_t = WithInt<4, float, ripple::strided_view_t>;
  using int_wrap_t = Params<ripple::Int64<4>, float, ripple::strided_view_t>;

  auto b1 = ripple::has_storage_layout_v<Normal>;
  auto b2 = ripple::has_storage_layout_v<mult_2_t>;
  auto b3 = ripple::has_storage_layout_v<mult_3_t>;
  auto b4 = ripple::has_storage_layout_v<default_t>;
  auto b5 = ripple::has_storage_layout_v<with_int_t>;
  auto b6 = ripple::has_storage_layout_v<int_wrap_t>;

  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);
  EXPECT_TRUE(b3);
  EXPECT_TRUE(b4);
  EXPECT_FALSE(b5);
  EXPECT_TRUE(b6);
}

TEST(storage_storage_traits, can_determine_storage_layout_type) {
  using mult_2_t   = Params<int, ripple::contiguous_view_t>;
  using mult_3_t   = Params<ripple::strided_view_t, int, float>;
  using default_t  = Default<int>;
  using with_int_t = WithInt<4, float, ripple::strided_view_t>;
  using int_wrap_t = Params<ripple::Int64<4>, ripple::contiguous_owned_t>;

  auto norm_none         = ripple::storage_layout_kind_v<Normal>;
  auto mult_contig_view  = ripple::storage_layout_kind_v<mult_2_t>;
  auto mult_strided_view = ripple::storage_layout_kind_v<mult_3_t>;
  auto def_contig_view   = ripple::storage_layout_kind_v<default_t>;
  auto ct_int_owned      = ripple::storage_layout_kind_v<int_wrap_t>;

  // Can't determine the type when there are non template type parameters.
  auto int_none = ripple::storage_layout_kind_v<with_int_t>;


  EXPECT_TRUE(norm_none         == ripple::LayoutKind::none            );
  EXPECT_TRUE(mult_contig_view  == ripple::LayoutKind::contiguous_view );
  EXPECT_TRUE(mult_strided_view == ripple::LayoutKind::strided_view    );
  EXPECT_TRUE(def_contig_view   == ripple::LayoutKind::contiguous_view );
  EXPECT_TRUE(int_none          == ripple::LayoutKind::none            );
  EXPECT_TRUE(ct_int_owned      == ripple::LayoutKind::contiguous_owned);
}

//==--- [descritptor traits] -----------------------------------------------==//

// Dummy class that implements the AutoLayout interface, which allows the
// differently laid out storage types to be generated at compile time.
template <typename T, typename S = ripple::contiguous_owned_t>
struct Test : ripple::StridableLayout<Test<T, S>> {
  using descriptor_t = ripple::StorageDescriptor<
    S, ripple::StorageElement<int, 3>, float
  >;
};

using test_strided_t = Test<int, ripple::strided_view_t>;
using test_contig_t  = Test<int, ripple::contiguous_view_t>;
using test_owned_t   = Test<int, ripple::contiguous_owned_t>;
/*
TEST(storage_storage_traits, can_get_alloc_traits_for_stridable_layout){
  // Get strided allocation traits:
  using traits_t = 
    typename ripple::layout_traits_t<test_contig_t>::
    template alloc_traits_t<ripple::LayoutKind::strided_view>;

  const auto ref_same = std::is_same_v<
    typename traits_t::ref_t, test_strided_t
  >; 
  const auto ref_diff = std::is_same_v<
    typename traits_t::ref_t, test_contig_t
  >; 
  const auto copy_same = std::is_same_v<
    typename traits_t::copy_t, test_owned_t
  >;

  EXPECT_TRUE(ref_same);
  EXPECT_FALSE(ref_diff);
  EXPECT_TRUE(copy_same);
}

TEST(storage_storage_traits, can_get_contig_alloc_traits_for_auto_layout) {
  // Get contiguous allocation traits:
  using traits_t = 
    typename ripple::layout_traits_t<test_contig_t>::
    template alloc_traits_t<ripple::LayoutKind::contiguous_view>;

  const auto ref_same = std::is_same_v<
    typename traits_t::ref_t, test_contig_t
  >; 
  const auto ref_diff = std::is_same_v<
    typename traits_t::ref_t, test_strided_t
  >; 
  const auto copy_same = std::is_same_v<
    typename traits_t::copy_t, test_owned_t
  >;

  EXPECT_TRUE(ref_same);
  EXPECT_FALSE(ref_diff);
  EXPECT_TRUE(copy_same);
};
*/

#endif // RIPPLE_TESTS_STORAGE_STORAGE_TRAITS_TESTS_HPP
