include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT DEFINED muzero_INSTALL_CMAKEDIR)
   set(muzero_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/muzero"
       CACHE STRING "Path to muzero CMake files")
endif ()

install(TARGETS muzero EXPORT muzero_Targets
        RUNTIME COMPONENT muzero_Runtime
        LIBRARY COMPONENT muzero_Runtime
        NAMELINK_COMPONENT muzero_Development
        ARCHIVE COMPONENT muzero_Development
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY "${muzero_SOURCE_DIR}/include/" 
        TYPE INCLUDE
        COMPONENT muzero_Development)

if (BUILD_SHARED_LIBS)
    set(type shared)
    message(STATUS "Building muzero shared lib")
else ()
    set(type static)
    message(STATUS "Building muzero static lib")
endif ()

install(EXPORT muzero_Targets
        DESTINATION "${muzero_INSTALL_CMAKEDIR}"
        NAMESPACE muzero::
        FILE muzero-${type}-targets.cmake
        COMPONENT muzero_Development)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/muzeroConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/muzeroConfig.cmake"
    INSTALL_DESTINATION "${muzero_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/muzeroConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/muzeroConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/muzeroConfigVersion.cmake"
        DESTINATION "${muzero_INSTALL_CMAKEDIR}"
        COMPONENT muzero_Development)
