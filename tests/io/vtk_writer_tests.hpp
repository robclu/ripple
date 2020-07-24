//==--- ripple/tests/io/vtk_writer_tests.hpp --------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vtk_writer_tests.hpp
/// \brief This file defines tests for the vtk writer.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_IO_VTK_WRITER_TESTS_HPP
#define RIPPLE_TESTS_IO_VTK_WRITER_TESTS_HPP

#include <ripple/core/io/vtk_writer.hpp>
#include <gtest/gtest.h>
#include <cstring>

auto this_file_path() -> std::string {
  std::string file_path = __FILE__;
  return file_path.substr(0, file_path.rfind("/"));
}

TEST(io_vtk_writer_tests, can_write_metadata_correctly) {
  std::string correct_file   = this_file_path() + "/metadata_test_correct.vtk";
  std::string test_file_base = "metadata_test";

  using writer_t = ripple::MultidimWriter;
  using dims_t   = typename writer_t::DimSizes;
  dims_t dims{10, 1, 0};

  std::unique_ptr<writer_t> writer =
    std::make_unique<ripple::VtkWriter>(test_file_base, "Metadata Test");

  writer->set_resolution(0.1);

  // Write the header:
  writer->open();
  writer->write_metadata(dims);
  writer->close();

  int   correct_length = 0, test_length = 0;
  char *correct_buf = nullptr, *test_buf = nullptr;

  // Compare the file against the header.
  std::ifstream correct(correct_file);
  std::ifstream test(test_file_base + ".vtk");

  if (correct) {
    correct.seekg(0, correct.end);
    correct_length = correct.tellg();
    correct.seekg(0, correct.beg);

    correct_buf = static_cast<char*>(malloc(correct_length));
    correct.read(correct_buf, correct_length);
  } else {
    EXPECT_TRUE(false);
  }

  if (test) {
    test.seekg(0, test.end);
    test_length = test.tellg();
    test.seekg(0, test.beg);

    test_buf = static_cast<char*>(malloc(test_length));
    test.read(test_buf, test_length);
  } else {
    EXPECT_TRUE(false);
  }
  EXPECT_EQ(test_length, correct_length);

  auto res = std::memcmp(test_buf, correct_buf, test_length);
  EXPECT_EQ(res, 0);

  correct.close();
  test.close();
  free(correct_buf);
  free(test_buf);
}

#endif // RIPPLE_TESTS_IO_VTK_WRITER_TESTS_HPP
