#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_concat.h"

#define CHECK_ACL(expr)                                                     \
    do {                                                                    \
        const auto status = (expr);                                         \
        if (status != ACL_SUCCESS) {                                        \
            std::fprintf(stderr, "%s failed: %d\n", #expr, int(status));    \
            return 1;                                                       \
        }                                                                   \
    } while (false)

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const std::vector<uint16_t> host0 = {
        0x3c00, 0x4000, 0x4200,
        0x4400, 0x4500, 0x4600
    };
    const std::vector<uint16_t> host1 = {
        0x4700, 0x4800,
        0x4880, 0x4900
    };
    const std::vector<uint16_t> expected = {
        0x3c00, 0x4000, 0x4200, 0x4700, 0x4800,
        0x4400, 0x4500, 0x4600, 0x4880, 0x4900
    };
    std::vector<uint16_t> actual(expected.size(), 0);

    void* dev0 = nullptr;
    void* dev1 = nullptr;
    void* devOut = nullptr;
    CHECK_ACL(aclrtMalloc(
        &dev0, host0.size() * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(
        &dev1, host1.size() * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(
        &devOut, actual.size() * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(
        dev0, host0.size() * sizeof(uint16_t),
        host0.data(), host0.size() * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        dev1, host1.size() * sizeof(uint16_t),
        host1.data(), host1.size() * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    const int64_t shape0[] = {2, 3};
    const int64_t stride0[] = {3, 1};
    const int64_t shape1[] = {2, 2};
    const int64_t stride1[] = {2, 1};
    const int64_t shapeOut[] = {2, 5};
    const int64_t strideOut[] = {5, 1};

    aclTensor* tensor0 = aclCreateTensor(
        shape0, 2, ACL_FLOAT16, stride0, 0, ACL_FORMAT_ND,
        shape0, 2, dev0);
    aclTensor* tensor1 = aclCreateTensor(
        shape1, 2, ACL_FLOAT16, stride1, 0, ACL_FORMAT_ND,
        shape1, 2, dev1);
    aclTensor* tensorOut = aclCreateTensor(
        shapeOut, 2, ACL_FLOAT16, strideOut, 0, ACL_FORMAT_ND,
        shapeOut, 2, devOut);
    if (tensor0 == nullptr || tensor1 == nullptr || tensorOut == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        return 1;
    }

    const aclTensor* inputArray[] = {tensor0, tensor1};
    aclTensorList* inputList = aclCreateTensorList(inputArray, 2);
    if (inputList == nullptr) {
        std::fprintf(stderr, "aclCreateTensorList failed\n");
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnConcatGetWorkspaceSize(
        inputList, 1, tensorOut, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(
            &workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnConcat(
        workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStreamWithTimeout(stream, 5000));
    CHECK_ACL(aclrtMemcpy(
        actual.data(), actual.size() * sizeof(uint16_t),
        devOut, actual.size() * sizeof(uint16_t),
        ACL_MEMCPY_DEVICE_TO_HOST));

    bool equal = actual == expected;
    std::printf("expected:");
    for (auto value : expected) {
        std::printf(" %04x", value);
    }
    std::printf("\nactual:  ");
    for (auto value : actual) {
        std::printf(" %04x", value);
    }
    std::printf("\n%s\n", equal ? "PASS" : "FAIL");

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensorList(inputList);
    aclDestroyTensor(tensor0);
    aclDestroyTensor(tensor1);
    aclDestroyTensor(tensorOut);
    aclrtFree(dev0);
    aclrtFree(dev1);
    aclrtFree(devOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return equal ? 0 : 2;
}
