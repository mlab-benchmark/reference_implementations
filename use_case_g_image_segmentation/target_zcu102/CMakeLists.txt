cmake_minimum_required(VERSION 3.18.2)

project(unet)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_library(unetLibraries src/unet.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(unetLibraries ${OpenCV_LIBS})
target_link_libraries(unetLibraries -lvart-runner )
target_link_libraries(unetLibraries -lglog)
target_link_libraries(unetLibraries -lxir)
target_link_libraries(unetLibraries -lunilog)
target_link_libraries(unetLibraries -lpthread)
add_executable(unet src/main.cpp)
target_link_libraries(unet unetLibraries)

