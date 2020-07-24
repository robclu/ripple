//==--- ripple/io/multidim_writer.hpp ---------------------- -*- C++ -*- ---==//
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
/// \brief This file defines an interface for writing data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_IO_MULTIDIM_WRITER_HPP
#define RIPPLE_IO_MULTIDIM_WRITER_HPP

#include "printable_element.hpp"
#include <ripple/core/container/tensor.hpp>
#include <string>
#include <vector>

namespace ripple {

/// The Writer class defines an interface for writing data to a file.
class MultidimWriter {
 public:
  /// Struct for the sizes of the dimensions to write.
  struct DimSizes {
    size_t x = 1; //!< Size of x dimension.
    size_t y = 1; //!< Size of x dimension.
    size_t z = 1; //!< Size of x dimension.
  };

  /// The type of the container for the element names.
  using element_names_t = std::vector<std::string>;

  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor.
  MultidimWriter() = default;

  /// Destructor.
  virtual ~MultidimWriter() = default;

  //==--- [methods] --------------------------------------------------------==//

  // clang-format off
  /// Writes the elements from the \p data for the tensor.
  ///
  /// This will cause a compile time error if the data dos not implement the
  /// Printable interface.
  ///
  /// \param  data          The data to write.
  /// \param  element_names The names of the elements to write.
  /// \param  args          Additional arguments for getting an element.
  /// \tparam T             The type of the tensor data.
  /// \tparam Dims          The number of dimensions for the tensor.
  /// \tparam Args          The type of additional arguments.
  template <typename T, size_t Dims, typename... Args>
  auto write(
    const Tensor<T, Dims>& data,
    const element_names_t& element_names,
    Args&&...              args) -> void {
    open();
    // clang-format on
    auto dims = get_dimension_sizes(data);
    write_metadata(dims);

    PrintableElement elem;
    for (auto& element : element_names) {
      const auto name = element.c_str();
      elem            = get_printable_element(data, name, 0, 0, 0, args...);
      write_element_header(elem);
      for (auto k : ripple::range(dims.z == 0 ? 1 : dims.z)) {
        for (auto j : ripple::range(dims.y)) {
          for (auto i : ripple::range(dims.x)) {
            elem = get_printable_element(data, name, i, j, k, args...);
            write_element(elem);
          }
        }
      }
      write_element_footer();
    }
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Sets the name of the data to write.
  /// \param name The name to set for the data.
  virtual auto set_name(std::string name) -> void = 0;

  /// Sets the resolution of the data.
  /// \param res The resolution for the data.
  virtual auto set_resolution(double res) -> void = 0;

  /// Writes the metadata for the writer, if there is any.
  /// \param sizes The sizes of the dimension to write.
  virtual auto write_metadata(const DimSizes& sizes) -> void = 0;

  /// Writes the element to the output file.
  /// \param element The printable element to write.
  virtual auto write_element(const PrintableElement& element) -> void = 0;

  /// Writes any necessary header data for an \p element.
  /// \param element The printable element to write the start data
  virtual auto
  write_element_header(const PrintableElement& element) -> void = 0;

  /// Writes a footer after all element data has been written.
  virtual auto write_element_footer() -> void = 0;

  /// Tries to open the file for the writer, using the base filename for the
  /// writer, appending the \p suffix to the base filename, as well as
  /// the .vtk extension, and then appending the result to the path.
  /// \param suffix The suffix to append to the base filename.
  /// \param path   Path to the directory for the file.
  virtual auto open(std::string suffix = "", std::string path = "") -> void = 0;

  /// Closes the file being written to.
  /// otherwise returning false.
  virtual auto close() -> void = 0;

 private:
  // clang-format off
  /// Gets the printable element from the \p tensor at the given indices and
  /// with the \p name.
  /// \param tensor The tensor to get the printable element from.
  /// \param name   The name of the element to get.
  /// \param i      The index of the element in the x dimension.
  /// \param j      The index of the element in the y dimension.
  /// \param k      The index of the element in the z dimension.
  /// \param args   Additional arguments for getting the element.
  /// \tparam T      The type of the tensor data.
  /// \tparam Dims   The number of dimensions in the tensor.
  /// \tparam Args   The types of additional arguments.
  template <typename T, size_t Dims, typename... Args>
  auto get_printable_element(
    const Tensor<T, Dims>& tensor,
    const char*            name,
    size_t                 i,
    size_t                 j,
    size_t                 k,
    Args&&...              args) const -> PrintableElement {
    // clang-format on
    if constexpr (Dims == 1) {
      return tensor(i)->printable_element(name, args...);
    } else if (Dims == 2) {
      return tensor(i, j)->printable_element(name, args...);
    } else {
      return tensor(i, j, k)->printable_element(name, args...);
    }
  }
  // clang-format on

  /// Get the sizes of the dimensions from the \p tensor.
  /// \param  tensor The tensor to get the dimension sizes from.
  /// \tparam T      The type of the data in the tensor.
  /// \tparam Dims   The number of dimensions in the tensor.
  template <typename T, size_t Dims>
  auto get_dimension_sizes(const Tensor<T, Dims>& tensor) const -> DimSizes {
    return DimSizes{Dims >= 1 ? tensor.size(dim_x) : 1,
                    Dims >= 2 ? tensor.size(dim_y) : 1,
                    Dims >= 3 ? tensor.size(dim_z) : 0};
  }
}; // namespace ripple

} // namespace ripple

#endif // RIPPLE_IO_MULTIDIM_WRITER_HPP
