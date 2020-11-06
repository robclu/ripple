//==--- ripple/benchmarks/levelset_element.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_element.hpp
/// \brief This file defines a levelset element class for benchmarks.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BENCHMARK_LEVELSET_ELEMENT_HPP
#define RIPPLE_BENCHMARK_LEVELSET_ELEMENT_HPP

#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/storage/struct_accessor.hpp>
#include <ripple/core/utility/portability.hpp>

enum class State : uint32_t { source = 0, converged = 1, updatable = 2 };

template <typename T, typename Layout = ripple::StridedView>
struct LevelsetElement
: public ripple::PolymorphicLayout<LevelsetElement<T, Layout>> {
  /// Defines the types to store for FIM. We need the actual type, and a boolean
  /// to represent if the cell has converged.
  using Descriptor = ripple::StorageDescriptor<Layout, T, State>;
  using Storage    = typename Descriptor::Storage;

  Storage storage;

  ripple_host_device LevelsetElement(const LevelsetElement& e) noexcept
  : storage(e.storage) {}

  ripple_host_device LevelsetElement(const Storage& s) : storage(s) {}

  // Overload of assignment operator. Copies the data only, not if the cell has
  // converged.
  ripple_host_device auto
  operator=(const LevelsetElement& other) -> LevelsetElement& {
    storage = other.storage;
    return *this;
  }

  // Overload of assignment operator. Copies the value of \p val into the value
  // being wrapped.
  ripple_host_device auto operator=(T val) -> LevelsetElement& {
    value() = val;
    return *this;
  }

  /// Returns the value of the type being wrapped.
  ripple_host_device auto value() -> T& {
    return storage.template get<0>();
  }

  /// Returns the value of the type being wrapped.
  ripple_host_device auto value() const -> const T& {
    return storage.template get<0>();
  }

  /// Returns if the cell has converged.
  ripple_host_device auto state() -> State& {
    return storage.template get<1>();
  }

  /// Returns if the cell has converged.
  ripple_host_device auto state() const -> const State& {
    return storage.template get<1>();
  }
};

#endif // RIPPLE_BENCHMARK_LEVELSET_ELEMENT_HPP
