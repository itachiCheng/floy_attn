#ifndef FLASH_ATTENTION_SCORE_PROTO_H_
#define FLASH_ATTENTION_SCORE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FlashAttentionScore)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .OPTIONAL_INPUT(real_shift, ge::TensorType::ALL())
    .OPTIONAL_INPUT(drop_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(padding_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(prefix, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_qlen, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_kvlen, ge::TensorType::ALL())
    .OPTIONAL_INPUT(q_start_idx, ge::TensorType::ALL())
    .OPTIONAL_INPUT(kv_start_idx, ge::TensorType::ALL())
    .OUTPUT(softmax_max, ge::TensorType::ALL())
    .OUTPUT(softmax_sum, ge::TensorType::ALL())
    .OUTPUT(softmax_out, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .ATTR(scale_value, Float, 1)
    .ATTR(keep_prob, Float, 1)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .OP_END_FACTORY_REG(FlashAttentionScore);

}

#endif
