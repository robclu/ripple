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

/**
 * The VtkWriter type implements the multidim writer interface to write data to
 * a file in the Vtk format.
 */
class VtkWriter final : public MultidimWriter {
 public:
  /// Defines the type of the dimension sizes.
  using DimSizes = MultidimWriter::DimSizes;

  /**
   * Constructor which sets the base name of the file to write, as well as the
   * name of the data, and the names of any elements to write.
   * \param  base_filename The base name of the file to write to.
   * \param  data_name     The name of the data.
   */
  VtkWriter(std::string base_filename, std::string data_name = "");

  /** Destructor which closes the file. */
  ~VtkWriter();

  /**
   * Copy constructor to create another writer from this writer. This copies
   * the parameters from the other writer, but __does not__ open or write
   * any data for the writer.
   * \param other The other writer to copy from.
   */
  VtkWriter(const VtkWriter& other) noexcept;

  /**
   * Clones the writer to create a new multidimensional writer.
   * \return A shared multidimensional writer.
   */
  auto clone() const noexcept -> std::shared_ptr<MultidimWriter> final override;

  /**
   * Sets the name of the data to write.
   * \param name The name to set for the data.
   */
  auto set_name(std::string name) noexcept -> void final override;

  /**
   * Sets the resolution of the data.
   * \param res The resolution for the data.
   */
  auto set_resolution(double res) noexcept -> void final override;

  /**
   * Writes the metadata for the file, which is the header and the indices.
   * \param dims The dimensions to write.
   */
  auto write_metadata(const DimSizes& dims) noexcept -> void final override;

  /**
   * Writes the element to the output file.
   * \param element The printable element to write.
   */
  auto write_element(const PrintableElement& element) noexcept
    -> void final override;

  /**
   * Writes any necessary header data for an element.
   * \param element The printable element to write the start data
   */
  auto write_element_header(const PrintableElement& element) noexcept
    -> void final override;

  /**
   * Writes a footer after all element data has been written.
   */
  auto write_element_footer() noexcept -> void final override;

  /**
   * Opens the file for the writer, using the base filename for the writer and
   * the suffix.
   * \param path   Path to the directory for the file.
   * \param suffix The suffix to append to the base filename.
   */
  auto
  open(std::string path = "", std::string suffix = "") -> void final override;

  /**
   * Close the file for writing.
   */
  auto close() -> void final override;

 private:
  std::ofstream ofstream_;              //!< Output stream to write to.
  std::string   base_filename_ = "";    //!< Output base name.
  std::string   name_          = "";    //!< Name of the data.
  double        res_           = 1.0;   //!< Resolution for the data.
  bool          ids_done_      = false; //!< If inidices have been written.
  bool          header_done_   = false; //!< If inidices have been written.

  /**
   * Gets the number of points to output.
   * \return The number of points given the dimension sizes.
   */
  auto num_points(const DimSizes& dims) const -> size_t;

  /**
   * Gets the number of cells to output.
   * \return The number of cells to output for the dimension sizes.
   */
  auto num_cells(const DimSizes& dims) const -> size_t;

  /**
   * Gets the string for the kind of the printable element.
   * \param kind The kind of the printable element.
   * \return The string for the kind to print.
   */
  auto
  get_kind_string(PrintableElement::AttributeKind kind) const -> std::string;

  /**
   * Writes the header for the VTK file format to the file.
   */
  auto write_header() -> void;

  /**
   * Writes the data for the indices to the file.
   */
  auto write_indices(const DimSizes& dims) -> void;
};

} // namespace ripple

#endif // RIPPLE_VIZ_IO_VTK_WRITER_HPP
