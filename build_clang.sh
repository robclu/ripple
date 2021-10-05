#!/bin/bash

cmake \
	-DCMAKE_CUDA_COMPILER=/opt/clang-13.0.0/bin/clang++ 		\
	-DCMAKE_CUDA_HOST_COMPILER=/opt/clang-13.0.0/bin/clang++  	\
	-DCUDA_PATH=/usr/local/cuda-11.4 				\
	-DCMAKE_BUILD_TYPE=Release 					\
	-DCUDA_ARCHS=86 						\
	-DCMAKE_CXX_COMPILER=g++-10 					\
	-DRIPPLE_BUILD_BENCHMARKS=ON 				 	\
	-DRIPPLE_BUILD_WITH_KOKKOS=OFF 					\
	..
