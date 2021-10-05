#!/bin/bash

#	-DCMAKE_CUDA_COMPILER=/opt/clang-13.0.0/bin/clang++ 	\
#	-DCMAKE_CUDA_HOST_COMPILER=/opt/clang-13.0.0/bin/clang++  	\

cmake \
	-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc 	\
	-DCUDA_PATH=/usr/local/cuda-11.4 			\
	-DCMAKE_BUILD_TYPE=Release 				\
	-DCUDA_ARCHS=80 					\
	-DCMAKE_CXX_COMPILER=mpicxx 					\
	-DRIPPLE_BUILD_BENCHMARKS=ON 				 	\
	-DCMAKE_CUDA_HOST_COMPILER=mpicxx  				\
	-DRIPPLE_BUILD_WITH_KOKKOS=On 					\
	-DKokkos_ENABLE_CUDA=On 					\
	-DCMAKE_CUDA_ARCHITECTURES=80 					\
	-DKokkos_ARCH_AMPERE86=Off 					\
	-DKokkos_ARCH_AMPERE80=On 					\
	..
