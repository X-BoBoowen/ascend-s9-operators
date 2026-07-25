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
* \file matmul_policy.h
* \brief
*/
#ifndef IMPL_MATMUL_POLICY_MATMUL_POLICY_H
#define IMPL_MATMUL_POLICY_MATMUL_POLICY_H

#include "../context/context.h"
#include "../feature_trait/matmul_feature_trait.h"
#include "../resource/cube_in_buffer/cube_in_buffer.h"
#include "../resource/cube_out_buffer/cube_out_buffer.h"
#include "../scheduler/bias/bias_scheduler.h"
#include "../scheduler/scheduler.h"
#include "../stage/copy_cube_in/copy_cube_in.h"
#include "../stage/copy_cube_out/copy_cube_out.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MatmulPolicy is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulPolicy is only for internal usage, does not support extension or customized specialization!
*/
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
struct MatmulPolicy
{
public:
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using Context = MatmulContext<IMPL, MM_CFG>;
    using CubeOutBuffer = AscendC::Impl::Detail::CubeOutBuffer<IMPL, L0cT, MM_CFG>;
    using CopyCubeOut = AscendC::Impl::Detail::CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using CopyCubeInA = AscendC::Impl::Detail::CopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = CopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = CubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = CubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Scheduler = MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using BatchScheduler = AscendC::Impl::Detail::BatchScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using BiasScheduler = AscendC::Impl::Detail::BiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

/*
    TrianUpperMatmulPolicy is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    TrianUpperMatmulPolicy is only for internal usage, does not support extension or customized specialization!
*/
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
struct TrianUpperMatmulPolicy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using Scheduler = MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TriangularMode::TRIANUPPER>;
};

/*
    TrianLowerMatmulPolicy is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    TrianLowerMatmulPolicy is only for internal usage, does not support extension or customized specialization!
*/
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
struct TrianLowerMatmulPolicy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using Scheduler = MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TriangularMode::TRIANLOWER>;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _MATMUL_POLICY_H_
