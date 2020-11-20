cmake_minimum_required(VERSION 3.0)
project(powell_timing)

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)

add_executable(powell_timing "src/powell_timing.cpp")
target_link_libraries(powell_timing ceres)