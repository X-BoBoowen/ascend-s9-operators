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
 * \file scheduler_intf.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_INTF_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_INTF_H

#include "../../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MatmulScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    TriangularMode TR_MODE = TriangularMode::TRIANUNDEF, typename = void>
class MatmulScheduler
{
    using DstT = typename C_TYPE::T;
public:
    /**
     * @description: Compute matmul
     * @param: gm: global tensor to store output
     * @param: enAtomic: atomic mode when get output
     * @param: enSequentialWrite: true if output results to the same basic block
     * @param: fakeMsg: only used in intrablockpartsum scene, true if current computation is fake intrablock scene
     * @return: void
     */
    __aicore__ inline void Schedule(const GlobalTensor<DstT>& gm, uint8_t enAtomic,
        bool enSequentialWrite, bool fakeMsg) {}
    /**
     * @description: Compute one block of matmul
     * @param: enPartialSum: true if current block is based on previous compute results
     * @return: bool: true if current block is still computing
     */
    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        return false;
    }

    /**
     * @description: Reset scheduler's status
     * @param: void
     * @return: void
     */
    __aicore__ inline void Reset() {}

    /**
     * @description: Get current block's output to local tensor
     * @param: co2Local: local tensor to store output
     * @param: enAtomic: atomic mode when get output
     * @param: enSequentialWrite: true if output results to the same basic block
     * @return: void
     */
    __aicore__ inline void GetResult(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0,
        bool enSequentialWrite = false) {}

    /**
     * @description: Get current block's output to local tensor
     * @param: gm: global tensor to store output
     * @param: enAtomic: atomic mode when get output
     * @param: enSequentialWrite: true if output results to the same basic block
     * @return: void
     */
    __aicore__ inline void GetResult(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false) {}

#if __CCE_AICORE__ < 220
    /**
     * @description: Get current block's output to local tensor and global tensor
     * @param: gm: global tensor to store output
     * @param: co2Local: local tensor to store output
     * @param: enAtomic: atomic mode when get output
     * @param: enSequentialWrite: true if output results to the same basic block
     * @return: void
     */
    __aicore__ inline void GetResult(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
        uint8_t enAtomic = 0, bool enSequentialWrite = false) {}
#endif
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif
