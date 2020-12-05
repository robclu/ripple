//==--- ripple/src/io/vtk_writer.cpp ----------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vtk_writer.cpp
/// \brief This file implements the vtk writer class.
//
//==------------------------------------------------------------------------==//

#include "../include/ripple/core/io/vtk_writer.hpp"

namespace ripple {

// clang-format off
// Defines the flags for opening the file to write.
constexpr auto flags              = std::ios::out | std::ios::trunc;
// Defines the version of the vtk format being used.
constexpr auto version_string     = "# vtk DataFile Version 3.0\n";
// Defines the output type string for the data file.
constexpr auto datatype_string    = "ASCII\n";
// Defines the dataset type for the output.
constexpr auto datasettype_string = "DATASET STRUCTURED_GRID\n";
// Defines the string for the for dimensions.
constexpr auto dim_string         = "DIMENSIONS ";
// Defines the string for the lookup table.
constexpr auto lookup_string      = "LOOKUP_TABLE default\n";
// Defines the string to specify that points are being output.
constexpr auto point_string       = "POINTS ";
// Defines the string to specify that cell data are being output.
constexpr auto cell_string        = "CELL_DATA ";
// Defines the string for scalar data.
constexpr auto scalar_string      = "SCALARS";
// Defines the string for vector data.
constexpr auto vector_string      = "VECTORS";
// Defines the string for the extension for VTK files.
constexpr auto extension_string   = ".vtk";
// The minimum value which can be written.
constexpr auto min_val            = 1.0e-32;

VtkWriter::VtkWriter(std::string base_filename, std::string name)
: base_filename_{base_filename}, name_{name} {}

VtkWriter::~VtkWriter() {
  close();
}

VtkWriter::VtkWriter(const VtkWriter& other) noexcept
: base_filename_{other.base_filename_}, name_{other.name_}, res_{other.res_} {}

auto VtkWriter::clone() const noexcept -> std::shared_ptr<MultidimWriter> {
  return make_writer<VtkWriter>(*this);
}

auto VtkWriter::set_name(std::string name) noexcept -> void {
  name_ = name;
}

auto VtkWriter::set_resolution(double res) noexcept -> void {
  res_ = res;
}

auto VtkWriter::write_metadata(const DimSizes& dims) noexcept -> void {
  write_header();
  write_indices(dims);
}

auto VtkWriter::write_element(const PrintableElement& element) noexcept
  -> void {
  for (const auto& v : element.values()) {
    ofstream_ << (std::abs(v) > min_val ? v : 0.0) << " ";
  }
  ofstream_ << "\n";
}

auto VtkWriter::write_element_header(const PrintableElement& element) noexcept
  -> void {
  ofstream_ << get_kind_string(element.kind) << " " << element.name
            << " double\n";
  if (element.kind == PrintableElement::AttributeKind::scalar) {
    ofstream_ << lookup_string;
  }
}

auto VtkWriter::write_element_footer() noexcept -> void {
  ofstream_ << "\n\n";
}

auto VtkWriter::open(std::string path, std::string suffix) -> void {
  if (ofstream_.is_open()) {
    return;
  }

  if (path.length() != 0) {
    path += "/";
  }
  const auto file = path + base_filename_ + suffix + extension_string;
  ofstream_.open(file, flags);
}

auto VtkWriter::close() -> void {
  if (!ofstream_.is_open()) {
    return;
  }
  ofstream_.close();
}

auto VtkWriter::num_points(const DimSizes& dims) const -> size_t {
  return (dims.x + 1) * (dims.y + 1) * (dims.z + 1);
}

auto VtkWriter::num_cells(const DimSizes& dims) const -> size_t {
  return dims.x * dims.y * (dims.z == 0 ? 1 : dims.z);
}

auto VtkWriter::get_kind_string(PrintableElement::AttributeKind kind) const
  -> std::string {
  switch (kind) {
    case PrintableElement::AttributeKind::scalar: return scalar_string;
    case PrintableElement::AttributeKind::vector: return vector_string;
    default: return std::string("INVALID ATTRIBUTE KIND ");
  }
}

auto VtkWriter::write_header() -> void {
  if (header_done_) {
    return;
  }

  ofstream_ << version_string << name_ << "\n"
            << datatype_string << datasettype_string;
  header_done_ = true;
}

auto VtkWriter::write_indices(const DimSizes& dims) -> void {
  if (ids_done_) {
    return;
  }
  // First write the dimension information for the dataset.
  ofstream_ << dim_string;
  ofstream_ << dims.x + 1 << " " << dims.y + 1 << " " << dims.z + 1 << "\n";
  ofstream_ << point_string << num_points(dims) << " double\n";

  for (auto k : ripple::range(dims.z + 1)) {
    for (auto j : ripple::range(dims.y + 1)) {
      for (auto i : ripple::range(dims.x + 1)) {
        const auto idx_z = res_ * k;
        const auto idx_y = res_ * j;
        const auto idx_x = res_ * i;
        ofstream_ << idx_x << " " << idx_y << " " << idx_z << "\n";
      }
    }
  }
  ofstream_ << "\n" << cell_string << num_cells(dims) << "\n";
  ids_done_ = true;
}

} // namespace ripple