/*==--- ripple/src/graph/node.cpp -------------------------- -*- C++ -*- ---==**
 *
 *                                 Ripple
 *
 *               Copyright (c) 2019, 2020, 2021 Rob Clucas
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==------------------------------------------------------------------------==//
 *
 * \file  node.cpp
 * \brief This file implements a Node class for a graph.
 *
 *==------------------------------------------------------------------------==*/

#include <ripple/core/graph/node.hpp>
#include <ripple/core/utility/range.hpp>

namespace ripple {

template <size_t Size>
auto NodeInfo::id_from_indices(const std::array<uint32_t, Size>& indices)
  -> uint64_t {
  using namespace math;
  static_assert(Size <= 3, "Node id only valid for up to 3 dimensions!");
  return Size == 1   ? indices[0]
         : Size == 2 ? hash_combine(indices[0], indices[1])
         : Size == 3
           ? hash_combine(indices[2], hash_combine(indices[0], indices[1]))
           : 0;
}

template <size_t Size>
auto NodeInfo::name_from_indices(
  const std::array<uint32_t, Size>& indices) noexcept -> NodeInfo::Name {
  Name name = Size == 0 ? "" : std::to_string(indices[0]);
  for (auto i : range(Size - 1)) {
    name += "_" + std::to_string(indices[i + 1]);
  }
  return name;
}

/*==--- [explicit specializations] -----------------------------------------==*/

template uint64_t NodeInfo::id_from_indices<1>(const std::array<uint32_t, 1>&);
template uint64_t NodeInfo::id_from_indices<2>(const std::array<uint32_t, 2>&);
template uint64_t NodeInfo::id_from_indices<3>(const std::array<uint32_t, 3>&);

template typename NodeInfo::Name
NodeInfo::name_from_indices<1>(const std::array<uint32_t, 1>&);
template typename NodeInfo::Name
NodeInfo::name_from_indices<2>(const std::array<uint32_t, 2>&);
template typename NodeInfo::Name
NodeInfo::name_from_indices<3>(const std::array<uint32_t, 3>&);

} // namespace ripple