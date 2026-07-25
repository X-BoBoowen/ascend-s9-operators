/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file matmul_utils.h
* \brief
*/

#ifndef IMPL_MATMUL_UTILS_MATMUL_UTILS_H
#define IMPL_MATMUL_UTILS_MATMUL_UTILS_H

#include "matmul_config_utils.h"
#include "matmul_type_def.h"

namespace AscendC {
struct DataCopyOutParams {
    __aicore__ DataCopyOutParams()
    {
        quantMode = 0;
        cBurstNum = 0;
        burstLen = 0;
        srcStride = 0;
        dstStride = 0;
        oriNSize = 0;
        enUnitFlag = false;
        quantScalar = 0;
        curM = 0;
        curN = 0;
    }
    __aicore__ DataCopyOutParams(const uint16_t count, const uint16_t len,
        const uint16_t srcStrideIn, const uint32_t dstStrideIn, const uint16_t nSize, const bool unitFlag,
        const int curMPos, const int curNPos)
    {
        cBurstNum = count;
        burstLen= len;
        srcStride = srcStrideIn;
        dstStride = dstStrideIn;
        oriNSize = nSize;
        enUnitFlag = unitFlag;
        curM = curMPos;
        curN = curNPos;
    }
    uint8_t quantMode = 0;
    uint16_t cBurstNum = 0;
    uint16_t burstLen = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    uint16_t oriNSize = 0;
    bool enUnitFlag = false;
    uint64_t quantScalar = 0;
    int curM = 0;
    int curN = 0;
    uint64_t cbufWorkspaceAddr = 0;
};

struct SplitParams
{
    int16_t axisL1Len;
    int16_t kAxisL1Len;
    int16_t axisL1Offset;
    int16_t kAxisL1Offset;
    int16_t axisL0Len;
};

struct BatchOffsetInfo
{
    int32_t modA;
    int32_t divisorA;
    int32_t alignA;
    int32_t modB;
    int32_t divisorB;
    int32_t alignB;
    int32_t modBias;
    int32_t divisorBias;
    int32_t alignBias;
    bool setBiasFlag {false};
};

struct BatchSchedulerContext
{
    int32_t offsetA;
    int32_t offsetB;
    int32_t offsetBias;
    uint32_t reduceGNum;
    bool isReduceG;
    SplitParams aL0Params;
    SplitParams bL0Params;
};
template <typename SrcT> __aicore__ inline constexpr int32_t GetC0Size()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return 8;
    } else if (sizeof(SrcT) == sizeof(int8_t)) {
        return 32;
    }
    return 16;
}

template <typename T, typename U>
constexpr bool IsSameTypeV = AscendC::IsSameType<T, U>::value;

template <typename T, typename... Others>
struct IsTypeOneOf {
    static constexpr bool value = false;
};

template <typename T, typename First, typename... Others>
struct IsTypeOneOf<T, First, Others...> {
    static constexpr bool value = IsSameTypeV<T, First> || IsTypeOneOf<T, Others...>::value;
};

template <typename T, typename... Others>
constexpr bool IsTypeOneOfV = IsTypeOneOf<T, Others...>::value;

struct CopyGMParams {
    int dstOffset { 0 };
    int baseUseN { 0 };
    int blockCount { 0 };
    int dstStride { 0 };
    bool isComputeLineByLine { false };
};

template <typename T>
const LocalTensor<T> NULL_TENSOR;

template <typename T>
const GlobalTensor<T> GLOBAL_NULL_TENSOR;

template <typename T> struct GetDstType {
    using Type = T;
};

template <> struct GetDstType<float> {
    using Type = float;
};

template <> struct GetDstType<half> {
    using Type = float;
};

template <> struct GetDstType<int8_t> {
    using Type = int32_t;
};

#if __CCE_AICORE__ >= 220
template <> struct GetDstType<bfloat16_t> {
    using Type = float;
};

template <> struct GetDstType<int4b_t> {
    using Type = int32_t;
};
#endif

template <typename>
struct IsGlobalTensor : falseType {};

template <typename T>
struct IsGlobalTensor<GlobalTensor<T>> : trueType {};

template <typename T>
constexpr bool IsGlobalTensorV = IsGlobalTensor<T>::value;

int32_t constexpr GetNdNzMask(CubeFormat dstFormat, CubeFormat srcFormat)
{
    if ((srcFormat == CubeFormat::ND) && (dstFormat == CubeFormat::NZ)) {
        return 1;
    } else if ((srcFormat == CubeFormat::NZ) && (dstFormat == CubeFormat::ND)) {
        return Impl::NZ_MASK_VAlUE;
    }
    return 0;
}

template <typename SrcT>
__aicore__ inline constexpr static int32_t AuxGetFactor()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return Impl::FLOAT_FACTOR;
    }
    return 1;
}

template <typename SrcT>
__aicore__ inline constexpr static int32_t AuxGetC0Size()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return Impl::B32_C0SIZE;
    } else if (IsSameType<SrcT, int8_t>::value) {
        return Impl::B8_C0SIZE;
    } else if (IsSameType<SrcT, int4b_t>::value) {
        return Impl::B4_C0SIZE;
    }
    return Impl::B16_C0SIZE;
}

template <typename T>
__aicore__ inline T CeilT(T num1, T num2)
{
    ASCENDC_ASSERT((num2 > 0),
        { KERNEL_LOG(KERNEL_ERROR, "num2 is %d , which should be larger than 0", num2); });
    return (num1 + num2 - 1) / num2;
}

template <typename T>
__aicore__ inline T CeilAlignT(T num1, T num2)
{
    ASCENDC_ASSERT((num2 > 0),
        { KERNEL_LOG(KERNEL_ERROR, "num2 is %d , which should be larger than 0", num2); });
    return CeilT(num1, num2) * num2;
}

#if __CCE_AICORE__ == 220
template <class T, class U>
__aicore__ inline void InitKfcClient(T &matmulClient, U *tiling, TPipe *tpipe, KfcCommClient *client, int instIdx,
    GM_ADDR workspace)
{
    ASSERT(workspace != nullptr && "workspace cannot be nullptr when InitKFC");
    ASSERT(instIdx >= 0);
    matmulClient.client = client;
    matmulClient.instIdx = instIdx;
    matmulClient.cubeTiling.SetTiling((TCubeTiling *)tiling);
    matmulClient.mmCntAddr_ = reinterpret_cast<__gm__ KfcMsg*>(GetMatmulIncAddr(workspace, GetBlockIdxImpl(), instIdx));
    matmulClient.InitStatic();
    matmulClient.devEvtID = instIdx * 2 + GetSubBlockIdxImpl();
}
#endif
__aicore__ constexpr bool PhyPosIsL1(TPosition pos)
{
    ASSERT(pos != TPosition::MAX);
    if (pos == TPosition::A1 || pos == TPosition::B1 ||
        pos == TPosition::SHM || pos == TPosition::TSCM) {
        return true;
    }
#if (__CCE_AICORE__ == 220 || __CCE_AICORE__ == 300)
    if (pos == TPosition::C1) {
        return true;
    }
#endif
    return false;
}

__aicore__ constexpr bool PhyPosIsUB(TPosition pos)
{
    ASSERT(pos != TPosition::MAX);
    if (pos == TPosition::GM || pos == TPosition::A1 || pos == TPosition::A2 ||
        pos == TPosition::B1 || pos == TPosition::B2 || pos == TPosition::CO1 ||
        pos == TPosition::SHM || pos == TPosition::TSCM) {
        return false;
    }
#if (__CCE_AICORE__ <= 200)
    if (pos == TPosition::C2) {
        return false;
    }
#elif (__CCE_AICORE__ == 220)
    if (pos == TPosition::C1 || pos == TPosition::C2 || pos == TPosition::CO2 ||
        pos == TPosition::C2PIPE2GM) {
        return false;
    }
#elif (__CCE_AICORE__ == 300)
    if (pos == TPosition::C1 || pos == TPosition::C2) {
        return false;
    }
#endif
    return true;
}

__aicore__ constexpr bool PhyPosIsGM(TPosition pos)
{
    ASSERT(pos != TPosition::MAX);
    if (pos == TPosition::GM) {
        return true;
    }
#if (__CCE_AICORE__ == 220)
    if (pos == TPosition::CO2) {
        return true;
    }
#endif
    return false;
}

template <bool AShare, bool BShare> __aicore__ __inline__ void SyncCubeWithVec()
{
    // Ensure that the Cube starts to process the message after receiving the
    // signals of V0 and V1 in the case of ABshare.
    // This is needed because only V0 will communicate with Cube for kfc during ABshare, to prevent
    // V1 lags far behind V0 then the Cube output is overwritten by the next Cube calculation triggered by V0
    // before being consumed by V1.
#if defined(__DAV_C220_CUBE__)
    if constexpr (AShare && BShare) {
        constexpr uint16_t eventID = 9U;
        WaitEvent(eventID);
        return;
    }
#elif defined(__DAV_C220_VEC__)
    if constexpr (AShare && BShare) {
        constexpr uint16_t eventID = 9U;
        NotifyEvent<PIPE_MTE3>(eventID);
        return;
    }
#else
#endif
}

template <typename T>
__aicore__ constexpr int32_t GetBitSize()
{
    if constexpr (std::is_arithmetic<T>::value) {
        return sizeof(T) * ONE_BYTE_BIT_SIZE;
    }
    if constexpr (IsSameTypeV<T, AscendC::int4b_t>) {
        return ONE_BYTE_BIT_SIZE / 2;
    }
    return ONE_BYTE_BIT_SIZE * 2;
}

template <typename T>
__aicore__ constexpr T CeilNoLog(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
__aicore__ constexpr T MaxValue(T t)
{
    return t;
}

template <typename T, typename ...Args>
__aicore__ constexpr T MaxValue(T t, Args... args)
{
    T maxValue = MaxValue(args...);
    return t > maxValue ? t : maxValue;
}

template <typename T>
__aicore__ constexpr T MinValue(T t)
{
    return t;
}

template <typename T, typename ...Args>
__aicore__ constexpr T MinValue(T t, Args... args)
{
    T minValue = MinValue(args...);
    return t < minValue ? t : minValue;
}

template <typename T>
__aicore__ constexpr T Align(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
__aicore__ constexpr T AlignDown(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 / num2) * num2;
}

template <typename T>
__aicore__ constexpr int32_t GetTypeSize()
{
    if constexpr (std::is_arithmetic<T>::value) {
        return sizeof(T);
    }
    if constexpr (IsSameTypeV<T, AscendC::int4b_t>) {
        return 1;
    }
    return 1;
}

template <typename T>
__aicore__ inline T Ceil(T num1, T num2)
{
    ASCENDC_ASSERT((num2 > 0),
        { KERNEL_LOG(KERNEL_ERROR, "num2 is %d , which should be larger than 0", num2); });
    return (num1 + num2 - 1) / num2;
}

template <typename T>
__aicore__ inline T CeilAlign(T num1, T num2)
{
    ASSERT(num2 > 0);
    return Ceil(num1, num2) * num2;
}

// It is invoked by the matmulV3 operator and cannot be removed at present
__aicore__ inline uint16_t CeilDiv(uint16_t num1, uint16_t num2)
{
    ASSERT(num2 > 0);
    return (num1 + num2 - 1) / num2;
}

// It is invoked by the matmulV3 operator and cannot be removed at present
__aicore__ inline uint16_t CeilAlign(uint16_t num1, uint16_t num2)
{
    ASSERT(num2 > 0);
    return CeilDiv(num1, num2) * num2;
}

template <typename T, const auto& MM_CFG>
__aicore__ inline constexpr bool IsL0ACache()
{
    return (ToMatmulConfig(MM_CFG).singleCoreK <= ToMatmulConfig(MM_CFG).basicK) &&
        (ToMatmulConfig(MM_CFG).singleCoreM <= ToMatmulConfig(MM_CFG).basicM);
}

template <typename T, const auto& MM_CFG>
__aicore__ inline constexpr bool IsL0BCache()
{
    if constexpr (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT) {
        return ToMatmulConfig(MM_CFG).basicK * ToMatmulConfig(MM_CFG).basicN * sizeof(T) * Impl::DB_FACTOR <= L0BUF_SIZE;
    } else {
        return ToMatmulConfig(MM_CFG).singleCoreK <= ToMatmulConfig(MM_CFG).basicK * Impl::DB_FACTOR;
    }
}

template <typename A_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsL0Cache()
{
    if constexpr ((!ToMatmulConfig(MM_CFG).doNorm && !ToMatmulConfig(MM_CFG).doMultiDataLoad) ||
        ToMatmulConfig(MM_CFG).intraBlockPartSum || A_TYPE::layout != LayoutMode::NONE ||
        ToMatmulConfig(MM_CFG).isA2B2Shared) {
        return false;
    }
    if constexpr (ToMatmulConfig(MM_CFG).doMultiDataLoad && ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT) {
        return false;
    }
    if constexpr (ToMatmulConfig(MM_CFG).singleCoreM <= 0 || ToMatmulConfig(MM_CFG).singleCoreN <= 0 ||
        ToMatmulConfig(MM_CFG).singleCoreK <= 0 || ToMatmulConfig(MM_CFG).basicM <= 0 ||
        ToMatmulConfig(MM_CFG).basicN <= 0 || ToMatmulConfig(MM_CFG).basicK <= 0) {
        return false;
    }
    return IsL0ACache<typename A_TYPE::T, MM_CFG>() && IsL0BCache<typename A_TYPE::T, MM_CFG>();
}

template <typename A_TYPE, const auto& MM_CFG>
constexpr bool isNormEnableScheduler = DoMatmulNorm(MM_CFG) && (A_TYPE::layout == LayoutMode::NONE)
                                   && !ToMatmulConfig(MM_CFG).intraBlockPartSum;

template <typename A_TYPE, const auto& MM_CFG>
constexpr bool isNormDisableScheduler = DoMatmulNorm(MM_CFG) && ((A_TYPE::layout != LayoutMode::NONE)
                                   || ToMatmulConfig(MM_CFG).intraBlockPartSum);

template <typename A_TYPE, const auto& MM_CFG>
constexpr bool IsBmmEnableScheduler = DoMatmulNorm(MM_CFG) &&
    ((A_TYPE::layout != LayoutMode::NONE && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LESS_THAN_L1) ||
    (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LARGE_THAN_L1) ||
    (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1));

template <const auto& MM_CFG>
constexpr bool IsBasicBlockEnable = DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG);

template <const auto& MM_CFG>
constexpr bool IsIntrablock = DoMatmulNorm(MM_CFG) && ToMatmulConfig(MM_CFG).intraBlockPartSum;

enum class TriangularMode {
    TRIANUNDEF = 0,
    TRIANUPPER,
    TRIANLOWER,
};

template <const auto& MM_CFG>
constexpr bool IsKdimReorderLoad = ToMatmulConfig(MM_CFG).enableKdimReorderLoad;

template <const auto& MM_CFG>
constexpr bool NormInitScene = DoMatmulNorm(MM_CFG) || DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG);

template <const auto& MM_CFG>
constexpr bool MdlInitScene = DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG);
} // namespace AscendC
#endif // _MATMUL_UTILS_H_
