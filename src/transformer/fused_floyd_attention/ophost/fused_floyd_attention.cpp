/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedFloydAttention);

const std::array<const aclTensor *, 3>
FusedFloydAttention(const aclTensor *query, const aclTensor *key_0, const aclTensor *value_0,
                    const aclTensor *key_1, const aclTensor *value_1,
                    const aclTensor *attenMaskOptional,
                    double scaleValueOptional, aclOpExecutor *executor)
{
    L0_DFX(FusedFloydAttention, query, key_0, value_0, key_1, value_1,
           attenMaskOptional, scaleValueOptional);

    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    // auto softmaxOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(FusedFloydAttention,
                           OP_INPUT(query, key_0, value_0, key_1, value_1,
                                    attenMaskOptional),
                           OP_OUTPUT(softmaxMaxOut, softmaxSumOut, attentionOutOut),
                           OP_ATTR(static_cast<float>(scaleValueOptional)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedFloydAttention InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(FusedFloydAttention,
                                OP_INPUT(query, key_0, value_0, key_1, value_1,
                                         attenMaskOptional),
                                OP_OUTPUT(softmaxMaxOut, softmaxSumOut, attentionOutOut),
                                OP_ATTR(static_cast<float>(scaleValueOptional)));
    return {softmaxMaxOut, softmaxSumOut, attentionOutOut};
}

} // namespace l0op