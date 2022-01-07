cmake_minimum_required (VERSION 3.12)

# Set flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ABSL_PROPAGATE_CXX_STD ON)

# Message out the current compiler flags
message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

project(muzero-cpp)

find_package(Torch REQUIRED)
# Need to temporary disable the prefix path for torch, as it has a protobuf which interferes here
set(CMAKE_PREFIX_PATH "")

# Testing
enable_testing()

# Build abseil and include for linking for all child targets
add_subdirectory(external/abseil-cpp)
link_libraries( 
    absl::base
    absl::flags
    absl::flags_parse
    absl::strings
    absl::synchronization
    absl::optional
    absl::time
)

# Build tensorboard_logger
set(TEMP_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
add_subdirectory(external/tensorboard_logger)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/tensorboard_logger/)
link_libraries(tensorboard_logger)

# Let child targets include from external libraries not needed to be built as libraries
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libnop/include)

# Build child targets
add_subdirectory(muzero-cpp)
add_subdirectory(tests)
add_subdirectory(examples)
