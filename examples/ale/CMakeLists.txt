find_package(ale CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)

add_executable(muzero_ale muzero_ale.cpp)
target_link_libraries(muzero_ale PRIVATE muzero)
target_link_libraries(muzero_ale PRIVATE ale::ale-lib ${OpenCV_LIBS})
target_link_libraries(muzero_ale
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
    )
target_link_libraries(muzero_ale PRIVATE $<IF:$<TARGET_EXISTS:SDL2_image::SDL2_image>,SDL2_image::SDL2_image,SDL2_image::SDL2_image-static>)

# Treat third-party headers as system headers so their warnings are not
# reported by this target's strict warning flags.
target_include_directories(muzero_ale SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

if(TARGET ale::ale-lib)
    get_target_property(_ale_include_dirs ale::ale-lib INTERFACE_INCLUDE_DIRECTORIES)
    if(_ale_include_dirs)
        target_include_directories(muzero_ale SYSTEM PRIVATE ${_ale_include_dirs})
    endif()
endif()

# With static LTO builds, GCC may emit ALE-originated string overflow warnings
# during the final executable link pass (WPA/LTRANS). Suppress at link time.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(muzero_ale PRIVATE -Wno-stringop-overflow)
endif()
