#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import os
import numpy as np
import sys
import torch
import torch_npu

case_name = sys.argv[1]
if case_name == 'test_flash_attention_score':
    with open(os.path.join(os.path.dirname(__file__), "layout.bin"), "r") as file:
        data = file.readlines()
        # print(data)
        line_id = int(data[0].strip())
        list_data = data[line_id].strip().split(" ")
        # print("list_data: ", list_data)
    B = int(list_data[0])
    S = int(list_data[1])
    N = int(list_data[2])
    D = int(list_data[3])
    is_float16 = int(list_data[4])
    if is_float16:
        dtype = np.float16
    else:
        dtype = np.bfloat16

    # query = np.random.uniform(-1, 1, (B, S, N, D)).astype(dtype)
    # key = np.random.uniform(-1, 1, (B, S, N, D)).astype(dtype)
    # value = np.random.uniform(-1, 1, (B, S, N, D)).astype(dtype)
    # bias = np.random.uniform(-1, 1, (1, N, S, S)).astype(dtype)



    query = np.random.uniform(-1, 1, (B, N, S, D)).astype(dtype)
    key = np.random.uniform(-1, 1, (B, N, S, D)).astype(dtype)
    value = np.random.uniform(-1, 1, (B, N, S, D)).astype(dtype)
    # bias = np.random.uniform(-1, 1, (1, N, S, S)).astype(dtype)
    mask = np.concatenate((np.ones((B, 1, S, S - 5)).astype(np.float32), np.zeros((B, 1, S, 5)).astype(np.float32)), axis=-1)

    evo_mask = 1 - mask.astype(np.uint8)
    # bias_mask = 1e12 * (mask - 1)
    # bias_mask = bias_mask.astype(dtype)
    # print("query.shape: ", query.shape)
    # print("query's dtype: ", query.dtype)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')
    # bias.tofile('bias.bin')
    evo_mask.tofile('atten_mask.bin')
    print("Generate Data Finished!")
    # query = Tensor(query)
    # key = Tensor(key)
    # value = Tensor(value)
    # # bias = Tensor(bias)
    # mask = Tensor(mask)
    # attention_mask = 1e12 * (mask - 1)
    # logits = ops.BatchMatMul(transpose_b=True)(query, key)
    # # print("This is the logits from basic ops", logits)
    # logits = ops.add(logits, attention_mask.astype(mstype.float16))
    # # logits = ops.add(logits, bias)
    # weight = ops.Softmax()(logits)
    # attn = ops.BatchMatMul()(weight, value)
    # attn = attn.asnumpy()
    device = torch.device('npu')
    q = torch.from_numpy(query).to(device)      # float16 -> NPU
    k = torch.from_numpy(key).to(device)
    v = torch.from_numpy(value).to(device)
    mask_npu = torch.from_numpy(mask).to(device)  # float32

    # ---------- 3. NPU 计算 ----------
    # 合并 batch 维，方便 bmm
    q2 = q.reshape(B * N, S, D)
    k2 = k.reshape(B * N, S, D)
    v2 = v.reshape(B * N, S, D)

    logits = torch.bmm(q2, k2.transpose(-2, -1))           # (B*N,S,S)
    logits = logits.reshape(B, N, S, S)

    attention_mask = 1e12 * (mask_npu - 1)                 # (B,1,S,S)
    logits = logits + attention_mask.half()                  # 按旧脚本用 float16 加

    weight = torch.softmax(logits.float(), dim=-1).half()    # 先 float32 稳数值，再降回 float16

    attn = torch.bmm(weight.reshape(B * N, S, S), v2)        # (B*N,S,D)
    attn = attn.reshape(B, N, S, D)

    # ---------- 4. 落盘结果 ----------
    attn.cpu().numpy().tofile('attn.bin')
    # attn.tofile('attn.bin')
   
    print("Generate Data Finished!")

elif case_name == 'test_fused_floyd_attention':
    B = 2
    H = 3
    N = 2
    M = 1152
    K = 1152
    D = 32
    query = np.random.uniform(-0.1, 0.1, (B, H, N, M, D)) # BHNMD
    key_0 = np.random.uniform(-0.1, 0.1, (B, H, N, K, D)) # BHNKD
    key_1 = np.random.uniform(-0.1, 0.1, (B, H, K, M, D)) # BHKMD
    value_0 = np.random.uniform(-0.1, 0.1, (B, H, N, K, D)) # BHNKD
    value_1 = np.random.uniform(-0.1, 0.1, (B, H, K, M, D)) # BHKMD
    mask = np.concatenate((np.ones((B, H, N, M, K//2)).astype(np.float32), np.zeros((B, H, N, M, K - K//2)).astype(np.float32)), axis=-1)

    attn_mask = 1 - mask.astype(np.uint8)
    query.tofile('query.bin')
    key_0.tofile('key_0.bin')
    key_1.tofile('key_1.bin')
    value_0.tofile('value_0.bin')
    value_1.tofile('value_1.bin')
    attn_mask.tofile('atten_mask.bin')
else:
    raise RuntimeError(f"Invalid case name:", case_name)