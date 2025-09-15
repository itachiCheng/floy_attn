/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FUSED_FLOYD_ATTENTION_H_
#define OP_API_INC_LEVEL2_ACLNN_FUSED_FLOYD_ATTENTION_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedFloydAttention的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnFusedFloydAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key0, const aclTensor *key1, const aclTensor *value0,
    const aclTensor *value1, const aclTensor *attenMaskOptional, float scaleValue, const aclTensor *softmaxMaxOut, 
    const aclTensor *softmaxSumOut, const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFusedFloydAttention的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFusedFloydAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FUSED_FLOYD_ATTENTION_H_
