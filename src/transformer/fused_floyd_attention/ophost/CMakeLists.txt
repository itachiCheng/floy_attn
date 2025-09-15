# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME FusedFloydAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnInner PRIVATE
        fused_floyd_attention_def.cpp
)

target_sources(opapi PRIVATE
        fused_floyd_attention.cpp
        aclnn_fused_floyd_attention.cpp
)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            fused_floyd_attention.cpp
            aclnn_fused_floyd_attention.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            fused_floyd_attention.cpp
            aclnn_fused_floyd_attention.cpp
    )
endif ()

target_sources(optiling PRIVATE
        fused_floyd_attention_tiling.cpp
        fused_floyd_attention_tiling_common.cpp
        fused_floyd_attention_tiling_general.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        # ${CMAKE_CURRENT_SOURCE_DIR}/../../flash_attention_score_grad/ophost
)

target_sources(opsproto PRIVATE
        fused_floyd_attention_proto.cpp
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_fused_floyd_attention.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
