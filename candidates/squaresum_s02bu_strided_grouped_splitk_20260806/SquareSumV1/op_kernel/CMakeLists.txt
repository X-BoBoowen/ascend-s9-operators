if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)
endif()

add_ops_compile_options(SquareSumV1 OPTIONS --tiling_key=1,2,3,4,5,6,7,8,9,10,11,12,13,14)

add_kernels_compile()

if (ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()
