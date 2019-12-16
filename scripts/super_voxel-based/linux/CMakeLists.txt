cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cloud_viewer)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(SOURCES sence_segmentation_rgb.cpp  segment_process.cpp file_trs.cpp file_trs.h segment_process.h)
add_executable (sence_segmentation_rgb ${SOURCES})

target_link_libraries (sence_segmentation_rgb ${PCL_LIBRARIES})
