cmake_minimum_required(VERSION 3.9)
project(miniC2D LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -static-libgcc -static-libstdc++ -std=c++17 -O2 -ffast-math -Wall -s -Wno-write-strings")

aux_source_directory(./src/ SOURCES)

add_executable(miniC2D ${SOURCES})

target_include_directories(miniC2D PUBLIC ./include/)
target_link_directories(miniC2D PUBLIC ./lib/)
target_link_libraries(miniC2D PUBLIC sat vtree nnf util gmp)
target_link_options(miniC2D PUBLIC -no-pie)
target_compile_definitions(miniC2D PUBLIC -DC2D_PACKAGE="miniC2D" -DC2D_VERSION="1.0.0" -DC2D_DATE="Sep 27, 2015")



