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
 * \file scheduler_mdl.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_H

#include "scheduler_intf.h"
#include "scheduler_mdl_base.h"
namespace AscendC {
namespace Impl {
namespace Detail {

/*
    MatmulScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    TriangularMode TR_MODE>
class MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE, enable_if_t<DoMatmulMDL(MM_CFG)>>
    : public MatmulMDLSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>
{
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE(PartialOutUtils);

    using TransT = typename A_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using DstT = typename C_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulMDLSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>;

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        MATMUL_MODULE(BiasScheduler)->SetBias(MATMUL_MODULE(BiasScheduler)->IsBias() && !enPartialSum);
        if (!BASE_MODULE::MoveNext()) {
            return false;
        }
        if (!enPartialSum) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        ReduceK(enPartialSum);
        return true;
    }

    __aicore__ inline void GetResult(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            GetResultImpl(gm, enAtomic, enSequentialWrite);
        } else {
            BASE_MODULE::GetResult(gm, enAtomic, enSequentialWrite);
        }
    }

    __aicore__ inline void GetResult(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            static_assert(ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT,
                "When scheduleType is OUTER_PRODUCT, GetResult function of LocalTensor output is not supported.");
        } else {
            BASE_MODULE::GetResult(co2Local, enAtomic, enSequentialWrite);
        }
    }

private:
    __aicore__ inline void ReduceK(bool enPartialSum)
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
        if (BASE_MODULE::IsMDLKFullLoad()) {
            // K outer loop only circulates once
            ReduceKOneIter(enPartialSum);
        } else {
            ReduceKMultiIter(enPartialSum);
        }
    }

    __aicore__ inline void ReduceKOneIter(bool enPartialSum)
    {
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::InitSplitAParams();
        SplitParams bL0Params = BASE_MODULE::InitSplitBParams();
        MATMUL_MODULE(KLoop)->OuterStart();
        // CopyIn
        LocalTensor<TransT> a1;
        LocalTensor<TransT> b1;
        BASE_MODULE::CopyIn(a1, b1);
        bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        SplitPrepareOneIter(isATranspose, isBTranspose, aL0Params, bL0Params);
        LocalTensor<BiasT> bias = SplitBias(bL0Params.axisL0Len);
        int32_t kL1Stride;
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            kL1Stride = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else {
            kL1Stride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
        }
        // start k inner loop
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
                if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                    ComputeMDB(a1, b1, bias, aL0Params, bL0Params, isATranspose, isBTranspose, !enPartialSum, true);
                } else if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
                    ComputeNDB(a1, b1, bias, aL0Params, bL0Params, isATranspose, isBTranspose, !enPartialSum, true);
                }
            } else {
                ComputeKDB(a1, b1, aL0Params, bL0Params, isATranspose, isBTranspose, !enPartialSum, true);
            }
            aL0Params.kAxisL1Offset += kL1Stride;
            bL0Params.kAxisL1Offset += kL1Stride;
        } while (MATMUL_MODULE(KLoop)->InnerNext());
        if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            MATMUL_MODULE(BiasScheduler)->Free(bias);
        }
        BASE_MODULE::DoPreloadAWait();
    }

    __aicore__ inline void ReduceKMultiIter(bool enPartialSum)
    {
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::InitSplitAParams();
        SplitParams bL0Params = BASE_MODULE::InitSplitBParams();
        bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // curKaOuterIdx and curKbOuterIdx are used to decide if left or right matrix need to clear its l1 buffer
        int32_t curKaOuterIdx = MATMUL_MODULE(KLoop)->GetOuterKaIdx();
        int32_t curKbOuterIdx = MATMUL_MODULE(KLoop)->GetOuterKbIdx();
        do {
            // CopyIn
            LocalTensor<TransT> a1;
            LocalTensor<TransT> b1;
            BASE_MODULE::CopyIn(a1, b1);
            LocalTensor<BiasT> bias = SplitBias(bL0Params.axisL0Len);
            Compute(a1, b1, bias, enPartialSum, isATranspose, isBTranspose, aL0Params, bL0Params);
            if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
                MATMUL_MODULE(BiasScheduler)->Free(bias);
            }
            BASE_MODULE::DoPreloadAWait();
            BASE_MODULE::ClearL1BufferCache(curKaOuterIdx, curKbOuterIdx);
        } while (MATMUL_MODULE(KLoop)->OuterNext());
        BASE_MODULE::ResetCopyInBuffer();
        MATMUL_MODULE(PartialOutUtils)->Destroy();
    }

    __aicore__ inline LocalTensor<BiasT> SplitBias(const int32_t dataLen)
    {
        if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            auto bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
            MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            // ORDER_N(M db): split load bias here, ORDER_M(N db): split load bias in each dbLoop
            if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, dataLen);
            }
            return bias;
        } else {
            auto bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
                MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, dataLen);
            MATMUL_MODULE(BiasScheduler)->Free(bias);
            return {};
        }
    }

    __aicore__ inline void Compute(const LocalTensor<TransT>& a1, const LocalTensor<TransT>& b1,
        LocalTensor<BiasT>& bias, const bool enPartialSum, const bool isATranspose, const bool isBTranspose,
        SplitParams& aL0Params, SplitParams& bL0Params)
    {
        // sL0CInit and sL0CLast are used for Split
        bool sL0CInit = false;
        bool sL0CLast = false;
        BASE_MODULE::SplitPrepare(enPartialSum, isATranspose, isBTranspose, aL0Params, bL0Params, sL0CInit, sL0CLast);
        // prepare for Split
        int32_t kL1Stride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
        // start k inner loop
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
                if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                    ComputeMDB(a1, b1, bias, aL0Params, bL0Params, isATranspose, isBTranspose, sL0CInit, sL0CLast);
                } else if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
                    ComputeNDB(a1, b1, bias, aL0Params, bL0Params, isATranspose, isBTranspose, sL0CInit, sL0CLast);
                }
            } else {
                ComputeKDB(a1, b1, aL0Params, bL0Params, isATranspose, isBTranspose, sL0CInit, sL0CLast);
                MATMUL_MODULE(PartialOutUtils)->CopyOut();
            }
            aL0Params.kAxisL1Offset += kL1Stride;
            bL0Params.kAxisL1Offset += kL1Stride;
        } while (MATMUL_MODULE(KLoop)->InnerNext());
    }

    __aicore__ inline void ComputeKDB(const LocalTensor<TransT>& a1, const LocalTensor<TransT>& b1,
        const SplitParams& aL0Params, const SplitParams& bL0Params,
        const bool isATranspose, const bool isBTranspose, const bool sL0CInit, const bool sL0CLast)
    {
        MATMUL_MODULE(TBufPoolL0)->Allocate();
        LocalTensor<TransT> a2 = BASE_MODULE::SplitA(a1, aL0Params, isATranspose);
        LocalTensor<TransT> b2 = BASE_MODULE::SplitB(b1, bL0Params, isBTranspose);
        MATMUL_MODULE(TBufPoolL0)->EnQue();
        MATMUL_MODULE(TBufPoolL0)->DeQue();
        BASE_MODULE::CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor(), a2, b2, aL0Params.axisL0Len,
            bL0Params.axisL0Len, MATMUL_MODULE(KLoop)->GetBaseShape(), isATranspose, isBTranspose, sL0CInit, sL0CLast);
        MATMUL_MODULE(TBufPoolL0)->Free();
        MATMUL_MODULE(BiasScheduler)->Free();
    }

    __aicore__ inline void ComputeMDB(const LocalTensor<TransT>& a1, const LocalTensor<TransT>& b1,
        LocalTensor<BiasT>& bias, SplitParams& aL0Params, SplitParams& bL0Params, bool isATranspose,
        bool isBTranspose, bool sL0CInit, bool sL0CLast)
    {
        // Split b2
        LocalTensor<TransT> b2 = BASE_MODULE::SplitB(b1, bL0Params, isBTranspose);
        uint32_t l0aDBLoop = MATMUL_MODULE(MLoop)->GetL0DBLoopNum();
        for (uint32_t idx = 0; idx < l0aDBLoop; ++idx) {
            // allocate L0 buffer
            MATMUL_MODULE(TBufPoolL0)->Allocate();
            // Split a2
            aL0Params.axisL1Offset += (idx * aL0Params.axisL0Len);
            LocalTensor<TransT> a2 = BASE_MODULE::SplitA(a1, aL0Params, isATranspose);
            MATMUL_MODULE(TBufPoolL0)->EnQue();
            MATMUL_MODULE(TBufPoolL0)->DeQue();
            // prepare params and compute
            BASE_MODULE::CubeCompute(
                MATMUL_MODULE(CubeOutBuffer)->GetTensor()[idx * aL0Params.axisL0Len * bL0Params.axisL0Len], a2, b2,
                aL0Params.axisL0Len, bL0Params.axisL0Len, MATMUL_MODULE(KLoop)->GetBaseShape(),
                isATranspose, isBTranspose, sL0CInit, sL0CLast);
            MATMUL_MODULE(TBufPoolL0)->Free();
        }
        MATMUL_MODULE(BiasScheduler)->Free();
    }

    __aicore__ inline void ComputeNDB(const LocalTensor<TransT>& a1, const LocalTensor<TransT>& b1, LocalTensor<BiasT>& bias,
        SplitParams& aL0Params, SplitParams& bL0Params, bool isATranspose, bool isBTranspose, bool sL0CInit, bool sL0CLast)
    {
        // Split a2
        LocalTensor<TransT> a2 = BASE_MODULE::SplitA(a1, aL0Params, isATranspose);
        uint32_t l0bDBLoop = MATMUL_MODULE(NLoop)->GetL0DBLoopNum();
        for (uint32_t idx = 0; idx < l0bDBLoop; ++idx) {
            // allocate L0 buffer
             MATMUL_MODULE(TBufPoolL0)->Allocate();
            // Split b2
            bL0Params.axisL1Offset += (idx * bL0Params.axisL0Len);
            LocalTensor<TransT> b2 = BASE_MODULE::SplitB(b1, bL0Params, isBTranspose);
            MATMUL_MODULE(TBufPoolL0)->EnQue();
            MATMUL_MODULE(TBufPoolL0)->DeQue();
            // load bias
            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, bL0Params.axisL0Len, idx * bL0Params.axisL0Len);
            // prepare params and compute
            BASE_MODULE::CubeCompute(
                MATMUL_MODULE(CubeOutBuffer)->GetTensor()[idx * aL0Params.axisL0Len * bL0Params.axisL0Len],
                a2, b2, aL0Params.axisL0Len, bL0Params.axisL0Len, MATMUL_MODULE(KLoop)->GetBaseShape(),
                isATranspose, isBTranspose, sL0CInit, sL0CLast);
            MATMUL_MODULE(TBufPoolL0)->Free();
            MATMUL_MODULE(BiasScheduler)->Free();
        }
    }

    __aicore__ inline void SplitPrepareOneIter(const bool isATranspose, const bool isBTranspose,
        SplitParams& aL0Params, SplitParams& bL0Params)
    {
        aL0Params.kAxisL1Offset = 0;
        bL0Params.kAxisL1Offset = 0;
        if constexpr (PhyPosIsL1(A_TYPE::pos) && IsStaticPaddingEnable(MM_CFG)) {
            aL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else if constexpr (PhyPosIsL1(A_TYPE::pos) && IsFullStaticTiling(MM_CFG)) {
            aL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() != -1 ?
                MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() :
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else {
            aL0Params.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeA() * BASE_MODULE::c0Size_;
        }
        if constexpr (PhyPosIsL1(B_TYPE::pos) && IsStaticPaddingEnable(MM_CFG)) {
            bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else if constexpr (PhyPosIsL1(B_TYPE::pos) && IsFullStaticTiling(MM_CFG)) {
            bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else {
            bL0Params.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeB() * BASE_MODULE::c0Size_;
        }
        MATMUL_MODULE(LoadToA2)->Prepare(isATranspose, aL0Params.kAxisL1Len, aL0Params.axisL1Len);
        MATMUL_MODULE(LoadToB2)->Prepare(isBTranspose, bL0Params.kAxisL1Len);
    }

    __aicore__ inline void GetResultImpl(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
    {
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        int32_t co1Offset = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        if (MATMUL_MODULE(NLoop)->GetL0DBLoopNum() > 1) { // Means L0 N db, need to excute twice FixpipeL0CToGm
            FixpipeL0CToGm(gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                enAtomic, enSequentialWrite);
            FixpipeL0CToGm(gm, co1Local[co1Offset], MATMUL_MODULE(MLoop)->GetInnerIdx(),
                MATMUL_MODULE(NLoop)->GetInnerIdx() + 1, enAtomic, enSequentialWrite);
        } else if (MATMUL_MODULE(MLoop)->GetL0DBLoopNum() > 1) { // Means L0 M db, need to excute twice FixpipeL0CToGm
            FixpipeL0CToGm(gm, co1Local,
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(), enAtomic, enSequentialWrite);
            FixpipeL0CToGm(gm, co1Local[co1Offset], MATMUL_MODULE(MLoop)->GetInnerIdx() + 1,
                MATMUL_MODULE(NLoop)->GetInnerIdx(), enAtomic, enSequentialWrite);
        } else {
            FixpipeL0CToGm(gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                enAtomic, enSequentialWrite);
        }
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

    __aicore__ inline void FixpipeL0CToGm(const GlobalTensor<DstT>& gm, const LocalTensor<L0cT>& co1Local,
        int32_t curM, int32_t curN, uint8_t enAtomic, bool enSequentialWrite)
    {
        if constexpr (C_TYPE::format != CubeFormat::ND && C_TYPE::format != CubeFormat::ND_ALIGN &&
            C_TYPE::format != CubeFormat::NZ) {
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "Data format of C matrix should be ND, ND_ALIGN or NZ."); });
        }
        if (enAtomic == ATOMIC_ADD) {
            SetAtomicAdd<DstT>();
        } else if (enAtomic == ATOMIC_MAX) {
            SetAtomicMax<DstT>();
        } else if (enAtomic == ATOMIC_MIN) {
            SetAtomicMin<DstT>();
        }
        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(gm, co1Local, curM, curN,
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(gm, co1Local, curM, curN,
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        }

        if (enAtomic != 0) {
            SetAtomicNone();
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif
