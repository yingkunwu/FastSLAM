cmake_minimum_required(VERSION 3.17)
add_definitions(-std=c++14)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(MCL)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_BUILD_TYPE Debug)
set(cvplot_DIR "${PROJECT_SOURCE_DIR}/cvplot/build/")

find_package(cvplot REQUIRED)
find_package(OpenCV REQUIRED)

message("openCV_FOUND: ${OpenCV_FOUND}")
message("cvplot_FOUND: ${cvplot_FOUND}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/include/")
include_directories("${PROJECT_SOURCE_DIR}/cvplot/include/cvplot/")


# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h

# Add project executable
add_executable(MCL ${project_SRCS})
target_link_libraries(MCL ${OpenCV_LIBS} cvplot)