//==--- ripple/io/vtk_writer.hpp --------------------------- -*- C++ -*- ---==//
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

#ifndef RIPPLE_IO_VTK_WRITER_HPP
#define RIPPLE_IO_VTK_WRITER_HPP

#include "multidim_writer.hpp"
#include "printable_element.hpp"
#include <ripple/core/utility/range.hpp>
#include <array>
#include <fstream>
#include <vector>

namespace ripple {

/// The VtkWriter type implements the writer interface to write data to a file
/// in the Vtk format.
class VtkWriter : public MultidimWriter {
  //==--- [constants] ------------------------------------------------------==//
  // clang-format off

  /// Defines the flags for opening the file to write.
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
  /// The minimum value which can be written.
  static constexpr auto min_val            = 1.0e-32;

  // clang-format on
 public:
  /// Defines the type of the dimension sizes.
  using dims_t = MultidimWriter::DimSizes;

  //==--- [construction] ---------------------------------------------------==//

  /// Constructor which sets the base name of the file to write, as well as the
  /// name of the data, and the names of any elements to write.
  /// \param  base_filename The base name of the file to write to.
  /// \param  name          The name of the data.
  VtkWriter(std::string base_filename, std::string name = "")
  : _base_filename{base_filename}, _name{name} {}

  /// Destructor which closes the file.
  ~VtkWriter() {
    close();
  }

  /// Copy constructor to create another writer from this writer. This copies
  /// the parameters from the other writer, but __does not__ open or write
  /// any data for the writer.
  VtkWriter(const VtkWriter& other) noexcept
  : _base_filename{other._base_filename},
    _name{other._name},
    _res{other._res} {}

  //==--- [interface] ------------------------------------------------------==//

  auto clone() const noexcept -> std::shared_ptr<MultidimWriter> override {
    return make_writer<VtkWriter>(*this);
  }

  /// Sets the name of the data to write.
  /// \param name The name to set for the data.
  auto set_name(std::string name) noexcept -> void override {
    _name = name;
  }

  /// Sets the resolution of the data.
  /// \param res The resolution for the data.
  auto set_resolution(double res) noexcept -> void override {
    _res = res;
  }

  /// Writes the metadata for the file, which is the header and the indices.
  auto write_metadata(const dims_t& dims) noexcept -> void override {
    write_header();
    write_indices(dims);
  }

  /// Writes the element to the output file.
  /// \param element The printable element to write.
  auto
  write_element(const PrintableElement& element) noexcept -> void override {
    for (const auto& v : element.values()) {
      _ofstream << (std::abs(v) > min_val ? v : 0.0) << " ";
    }
    _ofstream << "\n";
  }

  /// Writes any necessary header data for an \p element.
  /// \param element The printable element to write the start data
  auto write_element_header(const PrintableElement& element) noexcept
    -> void override {
    _ofstream << get_kind_string(element.kind) << " " << element.name
              << " double\n";
    if (element.kind == PrintableElement::AttributeKind::scalar) {
      _ofstream << lookup_string;
    }
  }

  /// Writes a footer after all element data has been written.
  auto write_element_footer() noexcept -> void override {
    _ofstream << "\n\n";
  }

  /// Tries to open the file for the writer, using the base filename for the
  /// writer, appending the \p suffix to the base filename, as well as
  /// the .vtk extension, and then appending the result to the path.
  /// \param path   Path to the directory for the file.
  /// \param suffix The suffix to append to the base filename.
  auto open(std::string path = "", std::string suffix = "") -> void override {
    if (_ofstream.is_open()) {
      return;
    }

    if (path.length() != 0) {
      path += "/";
    }
    const auto file = path + _base_filename + suffix + extension_string;
    _ofstream.open(file, flags);
  }

  /// Tries to close the file, returning true on success, or if already closed,
  /// otherwise returning false.
  auto close() -> void override {
    if (!_ofstream.is_open()) {
      return;
    }
    _ofstream.close();
  }

 private:
  std::ofstream _ofstream;              //!< Output stream to write to.
  std::string   _base_filename = "";    //!< Output base name.
  std::string   _name          = "";    //!< Name of the data.
  double        _res           = 1.0;   //!< Resolution for the data.
  bool          _ids_done      = false; //!< If inidices have been written.
  bool          _header_done   = false; //!< If inidices have been written.

  //==--- [sizes] ----------------------------------------------------------==//

  /// Returns the number of points to output.
  auto num_points(const dims_t& dims) const -> size_t {
    return (dims.x + 1) * (dims.y + 1) * (dims.z + 1);
  }

  /// Returns the number of cells to output.
  auto num_cells(const dims_t& dims) const -> size_t {
    return dims.x * dims.y * (dims.z == 0 ? 1 : dims.z);
  }

  //==--- [get string for kind] --------------------------------------------==//

  /// Returns the string for the kind of the printable element.
  /// \param kind The kind of the printable element.
  auto
  get_kind_string(PrintableElement::AttributeKind kind) const -> std::string {
    switch (kind) {
      case PrintableElement::AttributeKind::scalar: return scalar_string;
      case PrintableElement::AttributeKind::vector: return vector_string;
      default: return std::string("INVALID ATTRIBUTE KIND ");
    }
  }

  //==--- [write functions] ------------------------------------------------==//

  /// Writes the header for the VTK file format to the file.
  auto write_header() -> void {
    if (_header_done) {
      return;
    }

    _ofstream << version_string << _name << "\n"
              << datatype_string << datasettype_string;
    _header_done = true;
  }

  /// Writes the data for the indices to the file.
  auto write_indices(const dims_t& dims) -> void {
    if (_ids_done) {
      return;
    }
    // First write the dimension information for the dataset.
    _ofstream << dim_string;
    _ofstream << dims.x + 1 << " " << dims.y + 1 << " " << dims.z + 1 << "\n";
    _ofstream << point_string << num_points(dims) << " double\n";

    for (auto k : ripple::range(dims.z + 1)) {
      for (auto j : ripple::range(dims.y + 1)) {
        for (auto i : ripple::range(dims.x + 1)) {
          const auto idx_z = _res * k;
          const auto idx_y = _res * j;
          const auto idx_x = _res * i;
          _ofstream << idx_x << " " << idx_y << " " << idx_z << "\n";
        }
      }
    }
    _ofstream << "\n" << cell_string << num_cells(dims) << "\n";
    _ids_done = true;
  }
};

} // namespace ripple

#endif // RIPPLE_VIZ_IO_VTK_WRITER_HPP
