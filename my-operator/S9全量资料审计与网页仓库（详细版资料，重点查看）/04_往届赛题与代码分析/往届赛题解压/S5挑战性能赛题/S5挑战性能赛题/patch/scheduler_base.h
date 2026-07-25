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
 * \file scheduler_base.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_BASE_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_BASE_H

#include "../../utils/matmul_module.h"
#include "../../utils/matmul_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/*
    MatmulSchedulerBase is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulSchedulerBase is only for internal usage, does not support extension or customized specialization!
*/

/*
    MatmulSchedulerBase is the base class for other specialized MatmulScheduler,
    it implements the common GetResult methods.
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    TriangularMode TR_MODE = TriangularMode::TRIANUNDEF, typename = void>
class MatmulSchedulerBase
{
public:
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);
    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using DstT = typename C_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;

    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe)
    {
        if constexpr (!NormInitScene<MM_CFG> && !MdlInitScene<MM_CFG> && !DoMatmulIBShareNorm(MM_CFG)) {
            ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Unsupported matmul version."); });
            return;
        }
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(cubeTiling);
        MATMUL_MODULE(MatmulShapeTiling)->template CheckTiling<SrcT, L0cT>();
        auto& var = MATMUL_PARAM_VAR;
        var.tpipe_ = tpipe;

#if __CCE_AICORE__ < 220 || __CCE_AICORE__ == 300
        MATMUL_MODULE(MatmulSubBlockInfo)->SetSubBlockIdx(0);
#endif

        auto shapeInfo = MATMUL_MODULE(MatmulShapeInfo);
        shapeInfo->InitParams();
        if (shapeInfo->GetSingleCoreM() > 0 && shapeInfo->GetSingleCoreN() > 0 && shapeInfo->GetSingleCoreK() > 0) {
            MATMUL_MODULE(MLoop)->Init(shapeInfo->GetSingleCoreM());
            MATMUL_MODULE(NLoop)->Init(shapeInfo->GetSingleCoreN());
            MATMUL_MODULE(KLoop)->Init(shapeInfo->GetSingleCoreK());
        }

        MATMUL_MODULE(TBufPoolL0)->Init();

        if constexpr (unlikely(Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsUnitFlagEnabled())) {
            SetMMLayoutTransform(0);
        }

        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        uint32_t shareUbSize = static_cast<uint32_t>(tiling.GetShareUbSize());
        // shareL1Size, shareL0CSize, shareUbSize
        uint32_t shareLens[SHARE_LEN_SIZE] = { static_cast<uint32_t>(tiling.GetShareL1Size()),
                                               static_cast<uint32_t>(tiling.GetShareL0CSize()), shareUbSize };
        InitShareBufStart(var.tpipe_, tiling.GetShareMode(), shareLens, SHARE_LEN_SIZE,
                          MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx());
        MATMUL_MODULE(CopyCubeInA)->Init();
        MATMUL_MODULE(CopyCubeInB)->Init();
        auto baseMN = IsBasicBlockEnable<MM_CFG> ? ToMatmulConfig(MM_CFG).basicM * ToMatmulConfig(MM_CFG).basicN
                                                 : tiling.GetBaseM() * tiling.GetBaseN();

        uint32_t lenFactor = 1;
#if __CCE_AICORE__ >= 220
        if constexpr (MdlInitScene<MM_CFG> && ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT) {
            lenFactor = DOUBLE_SIZE;
        }
#endif
        MATMUL_MODULE(CubeOutBuffer)->Init(baseMN, lenFactor);
        if constexpr (NormInitScene<MM_CFG>) {
#if __CCE_AICORE__ >= 220
            MATMUL_MODULE(BiasScheduler)->Init();
#endif
        } else {
            MATMUL_MODULE(BiasScheduler)->Init();
        }
        MATMUL_MODULE(MatmulQuantProcessor)->Init(tiling.GetBaseN());
        if constexpr (MdlInitScene<MM_CFG>) {
#if __CCE_AICORE__ < 200
            var.tpipe_->InitBuffer(var.qidA2_, 1, L0ASize_);
            var.tpipe_->InitBuffer(var.qidB2_, 1, L0BSize_);
#endif
        }
        InitShareBufEnd(var.tpipe_);
    }

    __aicore__ inline void End() {
        if constexpr (!NormInitScene<MM_CFG> && !MdlInitScene<MM_CFG> && !DoMatmulIBShareNorm(MM_CFG)) {
            ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Unsupported matmul version."); });
            return;
        }
        MATMUL_MODULE(CopyCubeInA)->Destroy();
        MATMUL_MODULE(CopyCubeInB)->Destroy();
        MATMUL_MODULE(BiasScheduler)->End();
        MATMUL_MODULE(TBufPoolL0)->ResetCache();
        MATMUL_MODULE(CubeOutBuffer)->Destroy();
        MATMUL_MODULE(MatmulQuantProcessor)->Destroy();
    }

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Matching error. This is an empty implementation."); });
        return false;
    }

    __aicore__ inline void Schedule(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite, bool fakeMsg) {}

    __aicore__ inline void Reset()
    {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Matching error. This is an empty implementation."); });
    }

    __aicore__ inline void GetResult(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        static_assert(ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT, "Unsupported scheduleType");
        GetResultImpl(co2Local, enAtomic, enSequentialWrite);
    }

    __aicore__ inline void GetResult(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        GetResultImpl(gm, enAtomic, enSequentialWrite);
    }

#if __CCE_AICORE__ < 220
    __aicore__ inline void GetResult(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
        uint8_t enAtomic = 0, bool enSequentialWrite = false) {
        static_assert(ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT, "Unsupported scheduleType");
        GetResultImpl(gm, co2Local, enAtomic, enSequentialWrite);
    }
#endif

protected:
    __aicore__ inline void  GetResultImpl(
        const LocalTensor<DstT>& co2Local, uint8_t enAtomic, bool enSequentialWrite)
    {
        (void)(enAtomic);
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        if constexpr (TR_MODE == TriangularMode::TRIANUPPER) {
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() > MATMUL_MODULE(NLoop)->GetInnerIdx()) {
                MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
                return;
            }
        } else if constexpr (TR_MODE == TriangularMode::TRIANLOWER) {
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() < MATMUL_MODULE(NLoop)->GetInnerIdx()) {
                MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
                return;
            }
        }
        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(co2Local, co1Local,
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(co2Local, co1Local,
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        }
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

#if __CCE_AICORE__ < 220
    __aicore__ inline void SetAtomic(uint8_t enAtomic)
    {
        if (enAtomic == ATOMIC_ADD) {
            SetAtomicAdd<DstT>();
        }
    }

    __aicore__ inline void  GetResultImpl(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
    {
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        SetAtomic(enAtomic);

        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(
                gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(
                gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        }

        ClearAtomic(enAtomic);
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

    __aicore__ inline void  GetResultImpl(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
        uint8_t enAtomic, bool enSequentialWrite)
    {
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        SetAtomic(enAtomic);

        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(gm, co2Local, co1Local,
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(gm, co2Local, co1Local,
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        }

        ClearAtomic(enAtomic);
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

#else
    __aicore__ inline void SetAtomic(uint8_t enAtomic)
    {
        if (enAtomic == ATOMIC_ADD) {
            SetAtomicAdd<DstT>();
        } else if (enAtomic == ATOMIC_MAX) {
            SetAtomicMax<DstT>();
        } else if (enAtomic == ATOMIC_MIN) {
            SetAtomicMin<DstT>();
        }
    }

    __aicore__ inline void  GetResultImpl(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
    {
        if constexpr (C_TYPE::format != CubeFormat::ND && C_TYPE::format != CubeFormat::ND_ALIGN &&
            C_TYPE::format != CubeFormat::NZ) {
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "Data format of C matrix should be ND, ND_ALIGN or NZ."); });
        }
        // remove dependency conflicts only for scene which is not db
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        SetAtomic(enAtomic);
        if constexpr (TR_MODE == TriangularMode::TRIANUPPER) {
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() > MATMUL_MODULE(NLoop)->GetInnerIdx()) {
                ClearAtomic(enAtomic);
                MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
                return;
            }
        } else if constexpr (TR_MODE == TriangularMode::TRIANLOWER) {
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() < MATMUL_MODULE(NLoop)->GetInnerIdx()) {
                ClearAtomic(enAtomic);
                MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
                return;
            }
        }

        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(
                gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(),MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(
                gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                MATMUL_MODULE(MLoop)->GetBaseBlockShape(),MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        }

        ClearAtomic(enAtomic);
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }
#endif

    __aicore__ inline void ClearAtomic(uint8_t enAtomic)
    {
        if (enAtomic != 0) {
            SetAtomicNone();
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif
