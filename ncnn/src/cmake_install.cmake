# Install script for directory: /home/ldavi/inference_fpga/ncnn/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ldavi/inference_fpga/ncnn/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/ldavi/inference_fpga/ncnn/src/libncnn.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/home/ldavi/inference_fpga/ncnn/src/allocator.h"
    "/home/ldavi/inference_fpga/ncnn/src/benchmark.h"
    "/home/ldavi/inference_fpga/ncnn/src/blob.h"
    "/home/ldavi/inference_fpga/ncnn/src/c_api.h"
    "/home/ldavi/inference_fpga/ncnn/src/command.h"
    "/home/ldavi/inference_fpga/ncnn/src/cpu.h"
    "/home/ldavi/inference_fpga/ncnn/src/datareader.h"
    "/home/ldavi/inference_fpga/ncnn/src/expression.h"
    "/home/ldavi/inference_fpga/ncnn/src/gpu.h"
    "/home/ldavi/inference_fpga/ncnn/src/layer.h"
    "/home/ldavi/inference_fpga/ncnn/src/layer_shader_type.h"
    "/home/ldavi/inference_fpga/ncnn/src/layer_type.h"
    "/home/ldavi/inference_fpga/ncnn/src/mat.h"
    "/home/ldavi/inference_fpga/ncnn/src/modelbin.h"
    "/home/ldavi/inference_fpga/ncnn/src/net.h"
    "/home/ldavi/inference_fpga/ncnn/src/option.h"
    "/home/ldavi/inference_fpga/ncnn/src/paramdict.h"
    "/home/ldavi/inference_fpga/ncnn/src/pipeline.h"
    "/home/ldavi/inference_fpga/ncnn/src/pipelinecache.h"
    "/home/ldavi/inference_fpga/ncnn/src/simpleocv.h"
    "/home/ldavi/inference_fpga/ncnn/src/simpleomp.h"
    "/home/ldavi/inference_fpga/ncnn/src/simplestl.h"
    "/home/ldavi/inference_fpga/ncnn/src/simplemath.h"
    "/home/ldavi/inference_fpga/ncnn/src/simplevk.h"
    "/home/ldavi/inference_fpga/ncnn/src/vulkan_header_fix.h"
    "/home/ldavi/inference_fpga/ncnn/src/ncnn_export.h"
    "/home/ldavi/inference_fpga/ncnn/src/layer_shader_type_enum.h"
    "/home/ldavi/inference_fpga/ncnn/src/layer_type_enum.h"
    "/home/ldavi/inference_fpga/ncnn/src/platform.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/home/ldavi/inference_fpga/ncnn/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/ldavi/inference_fpga/ncnn/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/ldavi/inference_fpga/ncnn/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES
    "/home/ldavi/inference_fpga/ncnn/src/ncnnConfig.cmake"
    "/home/ldavi/inference_fpga/ncnn/src/ncnnConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/ldavi/inference_fpga/ncnn/src/ncnn.pc")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/ldavi/inference_fpga/ncnn/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
