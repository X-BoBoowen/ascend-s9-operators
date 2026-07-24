if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)
endif()

add_kernel_compile(IndexAddFast ${CMAKE_CURRENT_SOURCE_DIR}/index_add.cpp)

if (ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()
