cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_cpp_benchmark)

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
