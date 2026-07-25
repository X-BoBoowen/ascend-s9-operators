/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scheduler_norm.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_H

#include "scheduler_intf.h"
#include "scheduler_norm_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <class A_TYPE, const auto& MM_CFG>
constexpr bool isSingleLargeBMM =
    A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1;
/*
    MatmulScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    TriangularMode TR_MODE>
class MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE,
    enable_if_t<(DoMatmulIBShareNorm(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> ||
    isSingleLargeBMM<A_TYPE, MM_CFG> || IsBasicBlockEnable<MM_CFG>) && !MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()>>
    : public MatmulNormSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>
{
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MatmulUnitFlag);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using TransT = typename A_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulNormSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>;

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        MATMUL_MODULE(BiasScheduler)->SetBias(MATMUL_MODULE(BiasScheduler)->IsBias() && !enPartialSum);
        if (!BASE_MODULE::MoveNext()) {
            return false;
        }
        if (!enPartialSum) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        Compute(enPartialSum);
        return true;
    }

    __aicore__ inline void Compute(bool enPartialSum = false)
    {
        if constexpr (TR_MODE == TriangularMode::TRIANUPPER) {
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() > MATMUL_MODULE(NLoop)->GetInnerIdx()) {
                return;
            }
        } else if constexpr (TR_MODE == TriangularMode::TRIANLOWER) {
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() < MATMUL_MODULE(NLoop)->GetInnerIdx()) {
                return;
            }
        }
        if constexpr (IsBasic(MM_CFG)) {
            // K outer loop only circulates once
            ComputeOneIter(enPartialSum);
        } else {
            // K outer loop only circulates multi-times
            ComputeMultiIter(enPartialSum);
        }
    }

private:
    __aicore__ inline void ComputeOneIter(bool enPartialSum)
    {
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::InitSplitAParams();
        SplitParams bL0Params = BASE_MODULE::InitSplitBParams();
        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // CopyIn
        LocalTensor<TransT> a1 = MATMUL_MODULE(CopyCubeInA)->LoadData(
            0, 0, MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeA());
        LocalTensor<TransT> b1 = MATMUL_MODULE(CopyCubeInB)->LoadData(
            0, 0, MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape());

        SplitBias(bL0Params.axisL0Len);

        // prepare for Split
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        // update some params in SplitParams which is related to k loop
        BASE_MODULE::SplitPrepare(isATranspose, isBTranspose, aL0Params, bL0Params);
        MATMUL_MODULE(TBufPoolL0)->Allocate();
        LocalTensor<TransT> a2 = SplitA(a1, aL0Params, isATranspose);
        LocalTensor<TransT> b2 = SplitB(b1, bL0Params, isBTranspose);
        MATMUL_MODULE(TBufPoolL0)->EnQue();
        MATMUL_MODULE(TBufPoolL0)->DeQue();

        // prepare params and compute
        CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor(), a2, b2, aL0Params.axisL0Len, bL0Params.axisL0Len,
            MATMUL_MODULE(KLoop)->GetTileShapeA(), isATranspose, isBTranspose, enPartialSum, !enPartialSum, true);

        MATMUL_MODULE(CopyCubeInA)->ClearLoadData(a1);
        MATMUL_MODULE(CopyCubeInB)->ClearLoadData(b1);
    }

    __aicore__ inline void ComputeMultiIter(bool enPartialSum)
    {
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::InitSplitAParams();
        SplitParams bL0Params = BASE_MODULE::InitSplitBParams();

        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            int32_t kOuterIdx = MATMUL_MODULE(KLoop)->GetOuterIdx();
            // CopyIn
            LocalTensor<TransT> a1 = MATMUL_MODULE(CopyCubeInA)->LoadData(
                MATMUL_MODULE(MLoop)->GetInnerIdx(), kOuterIdx,
                MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeA());
            LocalTensor<TransT> b1 = MATMUL_MODULE(CopyCubeInB)->LoadData(
                kOuterIdx, MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape());
            // update some params in SplitParams which is related to k loop
            bool sL0CInit = false;
            bool sL0CLast = false;
            BASE_MODULE::UpdateComputeParams(enPartialSum, sL0CInit, sL0CLast);
            SplitBias(bL0Params.axisL0Len);
            bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
            bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
            BASE_MODULE::SplitPrepare(isATranspose, isBTranspose, aL0Params, bL0Params);
            // allocate L0 buffer
            MATMUL_MODULE(TBufPoolL0)->Allocate();
            LocalTensor<TransT> a2 = SplitA(a1, aL0Params, isATranspose);
            LocalTensor<TransT> b2 = SplitB(b1, bL0Params, isBTranspose);
            MATMUL_MODULE(TBufPoolL0)->EnQue();
            MATMUL_MODULE(TBufPoolL0)->DeQue();
            // prepare params and compute
            CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor(), a2, b2, aL0Params.axisL0Len, bL0Params.axisL0Len,
                MATMUL_MODULE(KLoop)->GetTileShapeA(), isATranspose, isBTranspose, enPartialSum, sL0CInit, sL0CLast);

            MATMUL_MODULE(CopyCubeInA)->ClearLoadData(a1, MATMUL_MODULE(MLoop)->GetInnerIdx(), kOuterIdx);
            MATMUL_MODULE(CopyCubeInB)->ClearLoadData(b1, kOuterIdx, MATMUL_MODULE(NLoop)->GetInnerIdx());
        } while (MATMUL_MODULE(KLoop)->OuterNext());
    }

private:
    __aicore__ inline void SplitBias(const int32_t dataLen)
    {
        LocalTensor<BiasT> bias;
        if constexpr (IsBasic(MM_CFG)) {
            bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape());
        } else {
            bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
                MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        }
        MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, dataLen);
        MATMUL_MODULE(BiasScheduler)->Free(bias);
    }

    __aicore__ inline LocalTensor<TransT> SplitA(const LocalTensor<TransT>& a1,
        const SplitParams& aL0Params, const bool isATranspose)
    {
        auto posA = MATMUL_MODULE(MLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() + MATMUL_MODULE(KLoop)->GetInnerIdx();
        LocalTensor<TransT> a2;
        // Split
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::A2>(posA))) {
            a2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, TransT>();
            MATMUL_MODULE(LoadToA2)->Load(a2, a1, aL0Params.axisL1Len, aL0Params.kAxisL1Len,
                aL0Params.axisL0Len, MATMUL_MODULE(KLoop)->GetTileShapeA(), aL0Params.axisL1Offset,
                aL0Params.kAxisL1Offset, isATranspose);
        } else {
            a2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, TransT>();
        }
        return a2;
    }

    __aicore__ inline LocalTensor<TransT> SplitB(const LocalTensor<TransT>& b1,
        const SplitParams& bL0Params, const bool isBTranspose)
    {
        auto posB = MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() + MATMUL_MODULE(KLoop)->GetInnerIdx();
        LocalTensor<TransT> b2;
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::B2>(posB))) {
            b2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, TransT>();
            MATMUL_MODULE(LoadToB2)->Load(b2, b1, bL0Params.axisL1Len, bL0Params.kAxisL1Len,
                bL0Params.axisL0Len, MATMUL_MODULE(KLoop)->GetTileShapeA(), bL0Params.axisL1Offset,
                bL0Params.kAxisL1Offset, isBTranspose);
        } else {
            b2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, TransT>();
        }
        return b2;
    }

    __aicore__ inline void CubeCompute(const LocalTensor<L0cT>& cMatrix, const LocalTensor<TransT>& a2,
        const LocalTensor<TransT>& b2, const uint16_t madM, const uint16_t madN, const uint16_t madK,
        const bool isATranspose, const bool isBTranspose, const bool enPartialSum, const bool sL0CInit, const bool sL0CLast)
    {
        uint8_t unitFlag = MATMUL_MODULE(MatmulUnitFlag)->GetUnitFlag(sL0CLast);
        bool isBias;
        bool cmatrixSource;
        bool cmatrixInitVal;
        BASE_MODULE::UpdateBiasParams(enPartialSum, sL0CInit, cmatrixSource, cmatrixInitVal, isBias);
        MATMUL_MODULE(MmadCompute)->Compute(cMatrix, a2, b2, madM, madK,
            madN, isATranspose, isBTranspose, unitFlag, cmatrixSource, cmatrixInitVal, isBias);
        MATMUL_MODULE(TBufPoolL0)->Free();
        // clear data in related buffers
        MATMUL_MODULE(BiasScheduler)->Free();
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif
