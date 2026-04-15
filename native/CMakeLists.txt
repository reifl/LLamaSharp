cmake_minimum_required(VERSION 3.14)
project(llamasharp-native)

# ── llama.cpp submodule ───────────────────────────────────────────────────────
# All options passed on the cmake command line (GGML_NATIVE, BUILD_SHARED_LIBS,
# GGML_CUDA, GGML_VULKAN, AVX flags, …) are forwarded automatically.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp llama_cpp_build)

# ── llama-jinja ──────────────────────────────────────────────────────────────
# Thin Jinja2 C-API wrapper around common_chat_templates_apply.
# Disabled for Android (NDK toolchain does not provide the needed C++ stdlib
# features used by the common library's Jinja2 engine at link time).
option(LLAMA_BUILD_JINJA "Build llama-jinja Jinja2 C-API wrapper" ON)

if(LLAMA_BUILD_JINJA AND NOT ANDROID)
    add_library(llama-jinja SHARED llama_jinja.cpp)

    target_compile_features(llama-jinja PRIVATE cxx_std_17)

    target_include_directories(llama-jinja PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor
    )

    target_link_libraries(llama-jinja PRIVATE llama common)

    if(UNIX AND NOT APPLE)
        target_link_options(llama-jinja PRIVATE "-Wl,-rpath,$ORIGIN")
    elseif(APPLE)
        set_target_properties(llama-jinja PROPERTIES
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endif()
endif()
