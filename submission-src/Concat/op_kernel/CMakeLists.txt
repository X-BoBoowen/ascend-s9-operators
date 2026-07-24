add_kernel_compile(ConcatFast ${CMAKE_CURRENT_SOURCE_DIR}/concat.cpp)

if (ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()
