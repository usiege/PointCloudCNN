cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cloud_viewer)

find_package(PCL 1.8 REQUIRED)

if (WIN32)
    set(BOOL_WIN32 ON)
    #为了防止编译过程中报错，详细参照 https://blog.csdn.net/is2120/article/details/6385304
	set(CMAKE_CXX_FLAGS "/EHsc ${CMAKE_CXX_FLAGS}")
else (WIN32)
    set(BOOL_WIN32 OFF)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif(WIN32)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(SOURCES sence_segmentation_rgb.cpp  segment_process.cpp file_trs.cpp file_trs.h segment_process.h)
add_executable (sence_segmentation_rgb ${SOURCES})
target_link_libraries (sence_segmentation_rgb ${PCL_LIBRARIES})
