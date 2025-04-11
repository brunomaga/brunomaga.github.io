cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_cpp_benchmark)

# Fix error "CMAKE_CUDA_ARCHITECTURES must be non-empty if set."
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES
	benchmark.cpp
	benchmark.h
	gptlite.cpp
	gptlite.h
	main.cpp)

add_executable(main ${SOURCE_FILES} )
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
