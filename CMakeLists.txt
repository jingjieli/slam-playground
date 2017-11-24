cmake_minimum_required(VERSION 2.8.11)
project(SLAM_PLAYGROUND)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT CMAKE_BUILD_TYPE)
  	message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  	set(CMAKE_BUILD_TYPE Release CACHE STRING
    	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    	FORCE)
else (NOT CMAKE_BUILD_TYPE)
  	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    	message("\n=================================================================================")
    	message("\n-- Build type: Debug. Performance will be terrible!")
    	message("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
    	message("\n=================================================================================")
  	endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
endif (NOT CMAKE_BUILD_TYPE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		message("Build Slam Playground with GNU ...")
		set (CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")

	elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		message("Build Slam Playground with Clang ...")
		set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    endif ()
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Include: ${OpenCV_INCLUDE_DIRS}")
message(STTAUS "OpenCV Libs ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Use Boost 1.63.0
set(Boost_INCLUDE_DIRS ${BOOST_DIR}/include/)
set(Boost_LIBRARIES 
		${BOOST_DIR}/lib/libboost_system.so
		${BOOST_DIR}/lib/libboost_filesystem.so
	)
message(STATUS "Boost Include: ${Boost_INCLUDE_DIRS}")
message(STTAUS "Boost Libs ${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIRS})

add_executable(slam_playground ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(slam_playground ${OpenCV_LIBS} ${Boost_LIBRARIES})