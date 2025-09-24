/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.




 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_floyd_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {

constexpr int FLA_SOFTMAXMAX_F32_DIM0SHAPE = 8;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;

ge::graphStatus InferShapeFusedFloydAttention(gert::InferShapeContext *context)
{
    OPS_LOG_I(context, "Enter FusedFloydAttention runtime infershape impl.");

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    auto attrs = context->GetAttrs();
    const auto *queryDesc = context->GetInputDesc(0);
    OPS_LOG_E_IF_NULL(context, queryDesc, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)

    int64_t shapeB = 1;
    int64_t shapeS = 1;
    int64_t shapeT = 0;

    // BNSD
    shapeB = queryShape->GetDim(0);
    // 2: BNSD中S的shape
    shapeS = queryShape->GetDim(DIM_NUM_2);

    auto headNum = queryShape->GetDim(1);
    // softmaxMax, fp32: (B, N, S, 8)
    gert::Shape *softmaxMaxShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, softmaxMaxShape, return ge::GRAPH_FAILED)

    // 0, 1, 2, 3, 4 : dim idx
    softmaxMaxShape->SetDimNum(DIM_NUM_4);
    softmaxMaxShape->SetDim(0, shapeB);
    softmaxMaxShape->SetDim(1, headNum);
    softmaxMaxShape->SetDim(DIM_NUM_2, shapeS);
    softmaxMaxShape->SetDim(DIM_NUM_3, FLA_SOFTMAXMAX_F32_DIM0SHAPE);

    // softmaxSum, shape same as softmaxMax
    gert::Shape *softmaxSumShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, softmaxSumShape, return ge::GRAPH_FAILED)
    *softmaxSumShape = *softmaxMaxShape;


    gert::Shape *attentionOutShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFusedFloydAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);
    // softmax_max, outidx:0
    context->SetOutputDataType(0, DT_FLOAT);
    // softmax_sum, outidx:1
    context->SetOutputDataType(1, DT_FLOAT);
    // attention_out, outidx:3
    context->SetOutputDataType(2, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedFloydAttention).InferShape(InferShapeFusedFloydAttention).InferDataType(InferDataTypeFusedFloydAttention);

} // namespace ops