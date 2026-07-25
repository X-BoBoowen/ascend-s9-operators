#ifndef COMMON_H
#define COMMON_H 

#include <type_traits>
#include <cstdint>

// # define OPEN_LOG

// constexpr int PRINT_LEVEL = 0;

constexpr int CORENUM = 40;
constexpr int BLOCK_BYTES_SIZE = 32;

constexpr int MAXDIM_2 = 3;
constexpr int MAXDIM = 8;

constexpr int ITER_BYTES_SIZE_0 = 8192;

constexpr int WRITEBACK = 1024;

constexpr int BUFFER_NUM = 2;

template <typename T>
using U = typename std::conditional<
    sizeof(T) == 2,
    int16_t,
    int32_t
>::type;



#endif