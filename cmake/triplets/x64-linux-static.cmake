set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../toolchains/linux-toolchain.cmake)
set(VCPKG_ENV_PASSTHROUGH "LIBTORCH_ROOT")

# Suppress noisy warnings when building third-party ports (e.g. ALE)
# under modern GCC toolchains.
set(VCPKG_C_FLAGS "${VCPKG_C_FLAGS} -Wno-stringop-overflow")
set(VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS} -Wno-stringop-overflow")
