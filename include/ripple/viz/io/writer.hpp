//==--- ripple/viz/io/writer.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  writer.hpp
/// \brief This file defines an implementation for a static interface for types
///        which can write data in different formats.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_VIZ_IO_WRITER_HPP
#define RIPPLE_VIZ_IO_WRITER_HPP

#include <ripple/core/utility/portability.hpp> 
#include <string>

namespace ripple::viz {

/// The Writer interface defines a static interface for types which can write
/// data in a certain format.
/// \tparam Impl The implementation type for the interface.
template <typename Impl>
class Writer {
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Tries to open the file for the writer, appending \p filename_extra to any
  /// pre-configured base filename for the writer.
  /// \param filename_extra An extra part to append to the filename.
  auto open(std::string filename_extra = "") -> void {
    impl()->open(std::move(filename_extra));
  }

  /// Tries to close the file, returning true on success, or if already closed,
  /// otherwise returning false.
  auto close() -> void {
    return impl()->close();
  }

  /// Sets the name of the data to write.
  /// \param name The name to set for the data.
  auto set_name(std::string name) {
    impl()->set_name(std::move(name));
  }

  /// Sets the resolution of the data.
  /// \param res The resolution for the data.
  auto set_resolution(double res) {
    impl()->set_resolution(res);
  }

  /// Sets the dimensions of the data to write.
  /// \param size_x The size of the x dimension.
  /// \param size_y The size of the y dimension.
  /// \param size_z The size of the z dimension.
  auto set_dimensions(size_t size_x, size_t size_y = 1, size_t size_z = 0)
  -> void {
    impl()->set_dimensions(size_x, size_y, size_z);
  }

  /// Returns true if the dimesions are valid (none of the dimensions are zero),
  /// otherwise returs true.
  auto dimensions_valid() const -> bool {
    return impl()->dimensions_valid();
  }

  //==--- [writing] --------------------------------------------------------==//
  
  /// Writes the metadata for the writer.
  auto write_metadata() -> void {
    impl()->write_metadata();
  }

  /// Writes the elements from the data which is iterated over by the \p
  /// iterator and which have been configured to be written by the writer.
  ///
  /// This will cause a compile time error if the data iterated over by the \p
  /// iterator does not implement the Printable interface, or if the \p iterator
  /// is not an iterator.
  ///
  /// \param  iterator The iterator to the data to write.
  /// \param  args     Additional arguments which may be required.
  /// \tparam Iterator The type of the iterator.
  /// \tparam Args     The types of the additional arguments.
  template <typename Iterator, typename... Args>
  auto write(Iterator&& iterator, Args&&... args) -> void {
    impl()->write(
      std::forward<Iterator>(iterator), std::forward<Args>(args)...
    );
  }
};  

} // namespace ripple::viz

#endif // RIPPLE_VIZ_IO_WRITER_HPP

