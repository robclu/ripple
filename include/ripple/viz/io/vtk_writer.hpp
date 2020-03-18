//==--- ripple/viz/io/vtk_writer.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vtk_writer.hpp
/// \brief This file implements the writer interface to write data to a file in
///        the vtk format.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_VIZ_IO_VTK_WRITER_HPP
#define RIPPLE_VIZ_IO_VTK_WRITER_HPP

#include "writer.hpp"
#include <ripple/viz/printable/printable_element.hpp>
#include <ripple/core/algorithm/unrolled_for.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/utility/dim.hpp>
#include <ripple/core/utility/range.hpp>
#include <array>
#include <fstream>
#include <vector>

namespace ripple::viz {

/// The VtkWriter type implements the writer interface to write data to a file
/// in the Vtk format.
class VtkWriter : public Writer<VtkWriter> {
  //==--- [constants] ------------------------------------------------------==//

  /// Defines the flags for opening the file to write. It needs to be writable
  /// and appendable/
  static constexpr auto flags              = std::ios::out | std::ios::trunc;
  /// Defines the version of the vtk format being used.
  static constexpr auto version_string     = "# vtk DataFile Version 3.0\n";
  /// Defines the output type string for the data file.
  static constexpr auto datatype_string    = "ASCII\n";
  /// Defines the dataset type for the output.
  static constexpr auto datasettype_string = "DATASET STRUCTURED_GRID\n";
  /// Defines the string for the for dimensions.
  static constexpr auto dim_string         = "DIMENSIONS ";
  /// Defines the string for the lookup table.
  static constexpr auto lookup_string      = "LOOKUP_TABLE default\n";
  /// Defines the string to specify that points are being output.
  static constexpr auto point_string       = "POINTS ";
  /// Defines the string to specify that cell data are being output.
  static constexpr auto cell_string        = "CELL_DATA ";
  /// Defines the string for scalar data.
  static constexpr auto scalar_string      = "SCALARS";
  /// Defines the string for vector data.
  static constexpr auto vector_string      = "VECTORS";
  /// Defines the string for the extension for VTK files.
  static constexpr auto extension_string   = ".vtk";

  ///==--- [aliases] -------------------------------------------------------==//
  
  /// The type of the container for the dimensions of the dataset to write.
  using dims_t     = std::array<size_t, 3>;
  /// The type of the container for the printable elements.
  using elements_t = std::vector<std::string>;

 public:
  //==--- [construction] ---------------------------------------------------==//
  
  /// Constructor which sets the base name of the file to write, as well as the
  /// name of the data, and the elements to write, if there are any.
  /// \param  filename_base The base name of the file to write to.
  /// \param  name          The name of the data.
  /// \param  elements      The names of the printable elements.
  /// \tparam Elements      The type of the elements.
  template <typename... Elements>
  VtkWriter(std::string filename_base, std::string name, Elements&&... elements) 
  : _elements{elements...},
    _filename_base(filename_base),
    _name{name},
    _dims{0, 0, 0} {}

  /// Destructor which closes the file if it's open.
  ~VtkWriter() {
    close();
  }

  //==--- [interface] ------------------------------------------------------==//
   
  /// Tries to open the file for the writer, using the base filename for the
  /// writer, appending the \p filename_extra to the base filename, as well as
  /// the .vtk extension.
  /// \param filename_extra An extra part to append to the filename.
  /// \param path           Path to the directory for the file.
  auto open(std::string filename_extra = "", std::string path = "") -> void {
    if (path.length()) {
      path += "/";
    }
    const auto file = path + _filename_base + filename_extra + extension_string;
    _ofstream.open(file, flags);
  }

  /// Tries to close the file, returning true on success, or if already closed,
  /// otherwise returning false.
  auto close() -> void {
    if (_ofstream.is_open()) {
      _ofstream.close();
    }
    _header_done = false;
    _ids_done    = false;
    _dims_set    = false;
  }

  /// Sets the name of the data to write.
  /// \param name The name to set for the data.
  auto set_name(std::string name) {
    _name = name;
  }

  /// Sets the resolution of the data.
  /// \param res The resolution for the data.
  auto set_resolution(double res) {
    _res = res;
  }

  /// Sets the dimensions of the data to write, which is the number of cells to
  /// output data for.
  /// \param size_x The size of the x dimension.
  /// \param size_y The size of the y dimension.
  /// \param size_z The size of the z dimension.
  auto set_dimensions(size_t size_x, size_t size_y = 0, size_t size_z = 0)
  -> void {
    if (_dims_set) {
      return;
    }

    _dims[0] = size_x; 
    _dims[1] = size_y;
    _dims[2] = size_z;
    _dims_set = true;
  }

  //==--- [write] ----------------------------------------------------------==//

  /// Writes the metadata for the file, which is the header and the indices.
  auto write_metadata() -> void {
    write_header();
    write_indices();
  }

  /// Writes the elements from the data which is iterated over by the \p
  /// iterator which have been configured for the writer.
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
    static_assert(is_iterator_v<Iterator>, "Can only write an iterator!");
    static_assert(
      is_printable_v<decltype(*iterator)>, 
      "Type to write must implement the Printable interface!"
    );

    // Need to set or check the dimensions:
    set_dimensions_from_iter(std::forward<Iterator>(iterator));
    check_dimensions(std::forward<Iterator>(iterator));
    write_metadata();

    for (auto& element : _elements) {
      write_element(iterator, element, std::forward<Args>(args)...);
    }
  }

 private:
  elements_t    _elements;             //!< The elements of the data to print.
  std::string   _filename_base;        //!< The base name of the file to write.
  std::string   _name;                 //!< The name of data to write.
  std::ofstream _ofstream;             //!< The stream to write the data to.
  dims_t        _dims;                 //!< The sizes of the dimensions.
  double        _res         = 1.0;    //!< The resolution of the data.
  bool          _ids_done    = false;  //!< If inidices have been written.
  bool          _header_done = false;  //!< If inidices have been written.
  bool          _dims_set    = false;  //!< If the dimensions have beens set.

  //==--- [sizes] ----------------------------------------------------------==//

  /// Returns the number of points to output.
  auto num_points() const -> size_t {
    return (_dims[0] + 1) * (_dims[1] + 1) * (_dims[2] + 1);
  }

  /// Returns the number of cells to output.
  auto num_cells() const -> size_t {
    return _dims[0] * _dims[1] * (_dims[2] == 0 ? 1 : _dims[2]);
  }

  //==--- [dimensions] -----------------------------------------------------==//
  
  /// Sets the dimensions for the writer from the iterator to write.
  /// \param  iterator The iterator to use to set the dimensions.
  /// \tparam Iterator The type of the iterator.
  template <typename Iterator>
  auto set_dimensions_from_iter(Iterator&& iterator) -> void {
    if (_dims_set) {
      return;
    }

    size_t d = 0;
    while (d < iterator.dimensions()) {
      _dims[d] = iterator.size(d);
      d++;
    }
    while (d < 3) {
      _dims[d] = (d == 1) ? 1 : 0;
      d++;
    }
    _dims_set = true;
  }

  /// Checks the dimensions for the writer from the iterator to write. This
  /// should only be called if the dimensions have been set, otherwise it
  /// returns false. If the dimension sizes of the \p iterator match those for
  /// the writer then this returns true, otherwise it returns false.
  /// \param  iterator The iterator to check the dimensions for.
  /// \tparam Iterator The type of the iterator.
  template <typename Iterator>
  auto check_dimensions(Iterator&& iterator) -> bool {
    if (!_dims_set) {
      return false;
    }

    size_t d = 0;
    for (auto d : range(iterator.dimensions())) {
      if (_dims[d] != iterator.size(d)) {
        return false;
      }
    }
    return true;
  }

  //==--- [get string for kind] --------------------------------------------==//

  /// Returns the string for the kind of the printable element.
  /// \param kind The kind of the printable element.
  auto get_kind_string(PrintableElement::AttributeKind kind) const 
  -> std::string {
    switch (kind) { 
      case PrintableElement::AttributeKind::scalar:
        return scalar_string;
      case PrintableElement::AttributeKind::vector:
        return vector_string;
      default:
        return std::string("INVALID ATTRIBUTE KIND ");
    }
  }

  //==--- [write functions] ------------------------------------------------==//

  /// Writes the header for the VTK file format to the file.
  auto write_header() -> void {
    if (_header_done) {
      return;
    }

    _ofstream <<
      version_string << _name << "\n" << datatype_string << datasettype_string;
    _header_done = true;
  }

  /// Writes the data for the indices to the file.
  auto write_indices() -> void {
    if (_ids_done) {
      return;
    }
    // First write the dimension information for the dataset.
    _ofstream << dim_string;
    for (auto& d : _dims) {
      _ofstream << d + 1 << " ";
    }
    _ofstream << "\n" << point_string << num_points() << " float\n";

    for (auto k : ripple::range(_dims[2] + 1)) {
      for (auto j : ripple::range(_dims[1] + 1)) {
        for (auto i : ripple::range(_dims[0] + 1)) {
          const auto idx_z = _res * k;
          const auto idx_y = _res * j;
          const auto idx_x = _res * i;
          _ofstream << idx_x << " " << idx_y << " " << idx_z << "\n";
        }
      }
    }
    _ofstream <<  "\n" << cell_string << num_cells() << "\n";
    _ids_done = true;
  }

  /// Writes the header for an element with the \p name, and \p type.
  /// \param type The type of the element to write.
  auto write_element_header(const PrintableElement& element)
  -> void {
    _ofstream << get_kind_string(element.kind()) << " "
              << element.name()                  << " float\n";
    if (element.kind() == PrintableElement::AttributeKind::scalar) {
      _ofstream << lookup_string;
    }
  }

  /// Writes an \p element from the \p iterator to the data file, if the data
  /// iterated over by the \p iterator has a \p element element which can be
  /// written.
  /// \param  iterator The iterator to the data to write.
  /// \param  element  The name of the element to write to the file.
  /// \param  args     Additional arguments which may be required.
  /// \tparam Iterator The type of the iterator.
  /// \tparam Args     The types of the additional arguments.
  template <typename Iterator, typename... Args>
  auto write_element(
    Iterator&&         iterator, 
    const std::string& element ,
    Args&&...          args
  ) -> void {
    const auto element_name = element.c_str();
    if (!iterator->has_printable_element(element_name)) {
      return;
    }

    if (iterator.size() != num_cells()) {
      // assert(false, "iterator does not match writer dimensions!");
      return;
    }

    auto it = iterator;
    write_element_header(it->printable_element(element_name, args...));

    // We need to write the data in the same order that the indices were
    // written, so we need to eplicitly choose the order of iteration over the
    // data:
    constexpr auto dims = iterator_traits_t<Iterator>::dimensions;
    for (auto k : ripple::range(_dims[2] == 0 ? 1 : _dims[2])) {
      for (auto j : ripple::range(_dims[1])) {
        for (auto i : ripple::range(_dims[0])) {
          it = iterator;
          if (dims == 3) {
            it.shift(ripple::dim_z, k);
          }
          if (dims >= 2) {
            it.shift(ripple::dim_y, j);
          }
          it.shift(ripple::dim_x, i);
          auto printable_elem = it->printable_element(element_name, args...);
          for (auto v : printable_elem.values()) {
            _ofstream << v << " ";
          }
        }
      }
    }  
    _ofstream << "\n\n";
  }
};

} // namespace ripple::viz

#endif // RIPPLE_VIZ_IO_VTK_WRITER_HPP
