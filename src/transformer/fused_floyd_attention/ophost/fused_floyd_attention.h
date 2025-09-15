/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_FUSED_FLOYD_ATTENTION_OP_H_
#define OP_API_INC_LEVEL0_OP_FUSED_FLOYD_ATTENTION_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 3>
FusedFloydAttention(const aclTensor *query, const aclTensor *key0, const aclTensor *key1,
                    const aclTensor *value0, const aclTensor *value1, const aclTensor *attenMaskOptional,
                    float scaleValue, aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_OP_FUSED_FLOYD_ATTENTION_OP_H_
