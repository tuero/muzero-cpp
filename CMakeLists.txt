cmake_minimum_required (VERSION 3.25)

project(muzero VERSION 2.0.0 LANGUAGES CXX)

# Build options
option(MUZERO_BUILD_EXAMPLES "Build muzero examples" OFF)
option(MUZERO_BUILD_TESTS "Build muzero tests" OFF)

# Let muzero_SHARED_LIBS override BUILD_SHARED_LIBS
if(DEFINED muzero_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${muzero_SHARED_LIBS}")
endif()

# CPP library
add_library(muzero)
add_library(muzero::muzero ALIAS muzero)
set_target_properties(muzero PROPERTIES
                      VERSION ${muzero_VERSION}
                      SOVERSION ${muzero_VERSION_MAJOR}
                      POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    muzero PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_compile_features(muzero PUBLIC cxx_std_20)

# Include the install rules if the user wanted them (included by default when top-level)
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)
option(muzero_INCLUDE_PACKAGING "Include packaging rules for muzero" "${is_top_level}")
if(muzero_INCLUDE_PACKAGING)
    add_subdirectory(packaging)
endif()

# Public headers
set(MUZERO_PUBLIC_HEADERS
    include/muzero/abstract_game.h
    include/muzero/config.h
    include/muzero/muzero.h
    include/muzero/types.h
)

# External libs from vcpkg (unconditional)
find_package(absl CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(tensorboard_logger CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Link external libs (unconditional)
target_link_libraries(muzero PUBLIC 
    absl::base 
    absl::flat_hash_map 
    absl::flat_hash_set 
    absl::hash
    absl::strings 
    absl::flags
    absl::flags_parse
    absl::optional
    absl::synchronization
    absl::time
)
target_link_libraries(muzero PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(muzero PRIVATE tensorboard_logger::tensorboard_logger)
target_link_libraries(muzero PUBLIC ${TORCH_LIBRARIES})


# Add headers to source listing
target_sources(muzero PRIVATE ${MUZERO_PUBLIC_HEADERS})

# Library source
add_subdirectory(src)

# Build examples
if (MUZERO_BUILD_EXAMPLES)
    message("MUZERO_BUILD_EXAMPLES=ON, building examples")
    add_subdirectory(examples)
else()
    message("MUZERO_BUILD_EXAMPLES=OFF, skipping building examples")
endif()


