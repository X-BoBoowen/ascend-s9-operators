/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file glu_jvp.cpp
 */
#include <iostream>
#include <cstdio>
#include "glu_jvp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GluJvpTilingData tiling;
    const auto runtime_attrs = context->GetAttrs();
    int dim = *(runtime_attrs->GetInt(0));
    const auto inputShape = context->GetInputTensor(1)->GetOriginShape();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    uint64_t l2_bw, hbm_bw;
    uint64_t l0a, l0b, l0c, l1, l2, ub, hbm;
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum_0 = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, l2_bw);
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::HBM, hbm_bw);
    auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::HBM, hbm);
    uint16_t mmInputDims[8];
    int nInputDims = inputShape.GetDimNum();
    int inputSize = 1;
    for (int i = 0; i < inputShape.GetDimNum(); i++) {
        mmInputDims[i] = inputShape.GetDim(i);
        inputSize *= mmInputDims[i];
    }
    if (dim < 0) {
        dim += nInputDims;
    }
    int KS = 1;
    for (int i = nInputDims - 1; i > dim; i--) {
        KS *= mmInputDims[i];
    }
    int HI = inputSize / KS / mmInputDims[dim];
    int J = mmInputDims[dim];
    tiling.set_HI(HI);
    tiling.set_KS(KS);
    tiling.set_J(J);
    uint32_t size = context->GetInputTensor(0)->GetShapeSize();
    uint32_t aivNum = 40;
    auto dt = context->GetInputTensor(0)->GetDataType();
    int DataTypeSize = 0;
    if (dt == ge::DT_FLOAT) {
        DataTypeSize = 4;
    } else {
        DataTypeSize = 2;
    }
    uint32_t totalBytes = size * DataTypeSize;
    aivNum = std::min(aivNum, static_cast<uint32_t>((totalBytes + static_cast<uint32_t>(1023)) / static_cast<uint32_t>(1 * 1024)));
    uint32_t blockSize = static_cast<uint32_t>(J * KS / 2 * DataTypeSize);
    uint32_t blockNum = (size * DataTypeSize + blockSize - 1) / blockSize;
    aivNum = std::min(aivNum, blockNum);
    uint32_t smallSize = blockNum;
    if (aivNum != 0) {
        smallSize = smallSize / aivNum;
    }
    smallSize *= blockSize / DataTypeSize;
    uint32_t incSize = blockSize / static_cast<uint32_t>(DataTypeSize);
    uint16_t formerNum = 0;
    if (aivNum != 0) {
        formerNum = static_cast<uint16_t>(blockNum % aivNum);
    }
    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);
    if (dt == ge::DT_FLOAT) {
        if (blockSize > static_cast<uint32_t>(4 * 5 * 1024) || blockSize < static_cast<uint32_t>(4 * 1 * 1024)) {
            context->SetTilingKey(1);
            context->SetBlockDim(aivNum);
        } else {
            context->SetTilingKey(4);
            context->SetBlockDim(aivNum);
        }
    } else if (dt == ge::DT_BF16) {
        context->SetTilingKey(2);
        context->SetBlockDim(aivNum);
    } else {
        context->SetTilingKey(3);
        context->SetBlockDim(1);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GluJvp : public OpDef {
public:
    explicit GluJvp(const char* name) : OpDef(name) {
        this->Input("glu_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("jvp_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GluJvp);
} // namespace ops
