#==--- ripple/cmake/ripple_cmake.cmake --------------------------------------==#
#
#                         Copyright (c) 2019 Rob Clucas.
#  
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#
#
# Description :   This file defines functions to create executables which can
#                 support compiling source files such that execution can occur
#                 on both the CPU and the GPU using cuda with either clang or
#                 nvcc as the cuda compiler. This is not currently possible with
#                 cmake.
#           
#==--------------------------------------------------------------------------==#

# Sets the DIRECTORIES to be the include directories for compilation.
function(ripple_include_directories DIRECTORIES)
  set(
    RIPPLE_INCLUDE_DIRS "${DIRECTORIES} ${ARGN}"
    CACHE FORCE "ripple include directories" FORCE
  )
endfunction()

# Appends the DIRECTORIES to the list of include directories for compiling.
function(ripple_include_directories_append DIRECTORIES)
  set(
    RIPPLE_INCLUDE_DIRS "${RIPPLE_INCLUDE_DIRS} ${DIRECTORIES} ${ARGN}"
    CACHE FORCE "ripple include directories" FORCE
  )
endfunction()

# Sets the list of library directories for compiling to DIRECTORIES.
function(ripple_library_directories DIRECTORIES)
  set(
    RIPPLE_LIBRARY_DIRS "${DIRECTORIES} ${ARGN}"
    CACHE FORCE "ripple library directories" FORCE
  )
endfunction()

# Adds the DEFINITIONS to the global list of definitions.
function(ripple_add_definitions DEFINITIONS)
  set(
    RIPPLE_GLOBAL_DEFINITIONS "${DEFINITIONS} ${ARGN}"
    CACHE FORCE "ripple global definitions" FORCE
  )
endfunction()

# Sets the target flags for the TARGET to the given FLAGS.
function(fluid_target_flags TARGET FLAGS)
  set(
    ${TARGET}_FLAGS "${FLAGS} ${ARGN}"
    CACHE FORCE "ripple target: ${TARGET}_FLAGS" FORCE
  )
endfunction()

# Sets the linker libraries for the TARGET to LINK_LIBS.
function(ripple_target_link_libraries TARGET LINK_LIBS)
  set(
    ${TARGET}_LINK_LIBS "${LINK_LIBS} ${ARGN}"
    CACHE FORCE "ripple link libraries: ${TARGET}_LINK_LIBS" FORCE
  )
endfunction()

# Adda and executable for the TARGET with the assosciated TARGET_FILE.
function(ripple_add_executable TARGET TARGET_FILE)
  set(
    RIPPLE_TARGET_LIST "${RIPPLE_TARGET_LIST} ${TARGET}"
    CACHE FORCE "ripple target list" FORCE
  )

  set(${TARGET}_FILE "${TARGET_FILE}" CACHE FORCE "target file" FORCE)
  set(${TARGET}_DEPENDENCIES "${ARGN}" CACHE FORCE "dependecies" FORCE)
endfunction()

# Creates all the defined targets.
function(ripple_create_all_targets)
  # Append -I to all include directories.
  separate_arguments(RIPPLE_INCLUDE_DIRS)
  foreach(ARG ${RIPPLE_INCLUDE_DIRS})
    set(TARGET_INCLUDE_DIRS "${TARGET_INCLUDE_DIRS} -I${ARG}")
  endforeach()

  # Append -L to all library directories
  separate_arguments(RIPPLE_LIBRARY_DIRS)
  foreach(ARG ${RIPPLE_LIBRARY_DIRS})
    set(TARGET_LIBRARY_DIRS "${TARGET_LIBRARY_DIRS} -L${ARG}")
  endforeach()

  # Separate the arguments into something that Cmake likes:
  if (RIPPLE_GLOBAL_DEFINITIONS)
    separate_arguments(RIPPLE_GLOBAL_DEFINITIONS)
  endif()
  if (CMAKE_CXX_FLAGS)
    separate_arguments(CMAKE_CXX_FLAGS)
  endif()
  if (CMAKE_CUDA_FLAGS)
    separate_arguments(CMAKE_CUDA_FLAGS)
  endif()
  if (TARGET_INCLUDE_DIRS)
    separate_arguments(TARGET_INCLUDE_DIRS)
  endif()
  if(TARGET_LIBRARY_DIRS)
    separate_arguments(TARGET_LIBRARY_DIRS)
  endif()

  separate_arguments(RIPPLE_TARGET_LIST)
  foreach(RIPPLE_TARGET ${RIPPLE_TARGET_LIST})
    # Remove some whitespace ...
    string(REGEX REPLACE " " "" RIPPLE_TARGET ${RIPPLE_TARGET})

    # Compile object file for test file:
    get_filename_component(TARGET_NAME ${${RIPPLE_TARGET}_FILE} NAME_WE)
    get_filename_component(TARGET_EXT  ${${RIPPLE_TARGET}_FILE} EXT)

    # Check if we are trying to compile for cuda:
    string(REGEX MATCH "cu" CUDA_REGEX ${TARGET_EXT})
    if (${TARGET_EXT} MATCHES "cu")
      set(CUDA_FILE TRUE)
    else()
      set(CUDA_FILE FALSE)
    endif()
    #string(COMPARE EQUAL "cu" ${CUDA_REGEX} CUDA_FILE)

    # Check the file type, and 
    if (CUDA_FILE)
      set(TARGET_COMPILER_TYPE         CUDA )
      set(TARGET_COMPILER_TYPE_STRING "CUDA")
    else()
      set(TARGET_COMPILER_TYPE         CXX  )
      set(TARGET_COMPILER_TYPE_STRING "CXX ")
    endif()

    set(TARGET_COMPILER ${RIPPLE_${TARGET_COMPILER_TYPE}_COMPILER})
    set(TARGET_FLAGS    ${CMAKE_${TARGET_COMPILER_TYPE}_FLAGS})
    separate_arguments(TARGET_FLAGS)
    message(
      "-- Creating ${TARGET_COMPILER_TYPE_STRING} target : ${RIPPLE_TARGET}"
    ) 

    # Again, make a list that Cmake likes ...
    if (${RIPPLE_TARGET}_FLAGS)
      separate_arguments(${RIPPLE_TARGET}_FLAGS)
    endif()
    if (${RIPPLE_TARGET}_LINK_LIBS)
      separate_arguments(${RIPPLE_TARGET}_LINK_LIBS)
    endif()

    # Compile object files for each of the dependencies
    foreach(FILE ${${RIPPLE_TARGET}_DEPENDENCIES})
      get_filename_component(DEPENDENCY_NAME ${FILE} NAME_WE)
      get_filename_component(DEPENDENCY_EXT  ${FILE} EXT)

      # Set the compiler for the dependency:
      # Check if we are trying to compile for cuda:
      string(REGEX MATCH "cu" DEP_CUDA_REGEX ${DEPENDENCY_EXT})
      string(COMPARE EQUAL "cu" ${DEP_CUDA_REGEX} DEP_CUDA_FILE)

      # Check the file type, and 
      if (${DEP_CUDA_FILE})
        set(COMP_TYPE         CUDA )
        set(COMP_TYPE_STRING "CUDA")
      else()
        set(COMP_TYPE        CXX   )
        set(COMP_TYPE_STRING "CXX ")
      endif()

      set(DEP_COMPILER ${CMAKE_${COMP_TYPE}_COMPILER})
      set(DEP_FLAGS    ${CMAKE_${COMP_TYPE}_FLAGS})
      separate_arguments(DEP_FLAGS)
      message(
        "  Creating ${COMP_TYPE_STRING} dependency : ${RIPPLE_TARGET}"
      )

      set(OBJECTS "${OBJECTS} ${DEPENDENCY_NAME}.o")
      add_custom_command(
        OUTPUT  ${DEPENDENCY_NAME}.o
        COMMAND ${DEP_COMPILER}
        ARGS    ${TARGET_INCLUDE_DIRS}
                ${DEP_FLAGS}
                ${RIPPLE_GLOBAL_DEFINITIONS}
                ${${RIPPLE_TARGET}_FLAGS}
                -c ${FILE}
                -o ${DEPENDENCY_NAME}.o
      )
    endforeach()
    separate_arguments(OBJECTS)

    set(OBJECT ${TARGET_NAME}.o)
    add_custom_command(
      OUTPUT  ${TARGET_NAME}.o
      COMMAND ${TARGET_COMPILER}
      ARGS    ${TARGET_INCLUDE_DIRS}
              ${TARGET_FLAGS}
              ${RIPPLE_GLOBAL_DEFINITIONS}
              ${${RIPPLE_TARGET}_FLAGS}
              -c ${${RIPPLE_TARGET}_FILE}
              -o ${TARGET_NAME}.o
      )

    # Create a target for the test:
    add_custom_target(
      ${RIPPLE_TARGET} ALL
      COMMAND ${TARGET_COMPILER}
              ${TARGET_INCLUDE_DIRS}
              ${TARGET_FLAGS}
              ${${RIPPLE_TARGET}_FLAGS}
              ${RIPPLE_GLOBAL_DEFINITIONS}
              -o ${RIPPLE_TARGET} ${OBJECT} ${OBJECTS}
              ${TARGET_LIBRARY_DIRS}
              ${${RIPPLE_TARGET}_LINK_LIBS}
      DEPENDS ${OBJECT} ${OBJECTS}
    )

    #install(
    #  PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${RIPPLE_TARGE`T}
    #  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    set(OBJECT)
    set(OBJECTS)
    message(
      "-- Created ${TARGET_COMPILER_TYPE_STRING} target : ${RIPPLE_TARGET}"
    )

    # Clean up:
    set(${RIPPLE_TARGET}_DEPENDENCIES "" CACHE FORCE "")
    set(${RIPPLE_TARGET}_FILE         "" CACHE FORCE "")
  endforeach()
  set(RIPPLE_TARGET_LIST "" CACHE FORCE "ripple target list" FORCE)
endfunction()

