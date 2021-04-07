/**=--- ripple/benchmarks/saxpy_cublas.cu ------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  saxpy_cublas.cu
 * \brief This file implements a saxpy bechhmark using cublas.
 *
 *==------------------------------------------------------------------------==*/

#include "saxpy.hpp"
#include <iostream>
#include <cublas_v2.h>
#include <ripple/utility/timer.hpp>

/*
 * This is a simple saxpy benchmark. Run as
 * ./saxpy_cublas <num_elements>
 */

int main(int argc, char** argv) {
  cublasStatus_t status;
  cublasHandle_t h      = nullptr;
  Real*          host_x = nullptr;
  Real*          host_y = nullptr;

  if (argc > 1) {
    elements = std::atol(argv[1]);
  }

  host_x = new Real[elements];
  host_y = new Real[elements];

  // cublasInit();
  if (auto res = cublasCreate(&h); res != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Error on cublas creation!\n";
    return 1;
  }

  for (int i = 0; i < elements; ++i) {
    host_x[i] = xval;
    host_y[i] = yval;
  }

  Real* dev_x = nullptr;
  Real* dev_y = nullptr;
  cudaMalloc((void**)&dev_x, elements * sizeof(Real));
  cudaMalloc((void**)&dev_y, elements * sizeof(Real));

  if (auto res =
        cublasSetVector(elements, sizeof(host_x[0]), host_x, 1, dev_x, 1);
      res != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Error on cublas set!\n";
    return 1;
  };
  if (auto res =
        cublasSetVector(elements, sizeof(host_y[0]), host_y, 1, dev_y, 1);
      res != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Error on cublas set!\n";
    return 1;
  }
  cudaDeviceSynchronize();

  ripple::Timer timer;
  cublasSaxpy(h, elements, &aval, dev_x, 1, dev_y, 1);
  cudaDeviceSynchronize();
  double elapsed = timer.elapsed_msec();

  std::cout << "Elements: " << elements << " : Time: " << elapsed << " ms\n";

  if (h) {
    cublasDestroy(h);
  }

  cudaFree(dev_y);
  cudaFree(dev_x);
  delete[] host_y;
  delete[] host_x;

  return 0;
}