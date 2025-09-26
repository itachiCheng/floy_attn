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

def tsoftmax(x):
     x_max = torch.amax(x, dim=-1, keepdims=True)
     x_sub = x.sub(x_max)
     y = torch.exp(x_sub)
     x_sum = y.sum(dim=-1, keepdims=True)
     ans = y.div(x_sum).half()
     return ans, x_max, x_sum


if case_name == 'test_fused_floyd_attention':
    with open(os.path.join(os.path.dirname(__file__), "layout.bin"), "r") as file:
        data = file.readlines()
        line_id = int(data[0].strip())
        list_data = data[line_id].strip().split(" ")
    # B = int(list_data[0])
    # S = int(list_data[1])
    # N = int(list_data[2])
    # D = int(list_data[3])
    # is_float16 = int(list_data[4])
    # if is_float16:
    #     dtype = np.float16
    # else:
    #     dtype = np.bfloat16
    # query = np.random.uniform(-1, 1, (B, N, S, D)).astype(dtype)
    # key = np.random.uniform(-1, 1, (B, N, S, D)).astype(dtype)
    # value = np.random.uniform(-1, 1, (B, N, S, D)).astype(dtype)
    # mask = np.concatenate((np.ones((B, 1, S, S - 5)).astype(np.float32), np.zeros((B, 1, S, 5)).astype(np.float32)), axis=-1)

    # evo_mask = 1 - mask.astype(np.uint8)
    # query.tofile('query.bin')
    # key.tofile('key.bin')
    # value.tofile('value.bin')
    # evo_mask.tofile('atten_mask.bin')
    # print("Generate Data Finished!")
    # device = torch.device('npu')
    # q = torch.from_numpy(query).to(device)      # float16 -> NPU
    # k = torch.from_numpy(key).to(device)
    # v = torch.from_numpy(value).to(device)
    # mask_npu = torch.from_numpy(mask).to(device)  # float32

    # q2 = q.reshape(B * N, S, D)
    # k2 = k.reshape(B * N, S, D)
    # v2 = v.reshape(B * N, S, D)

    # logits = torch.bmm(q2, k2.transpose(-2, -1))           # (B*N,S,S)
    # logits = logits.reshape(B, N, S, S)

    # attention_mask = 1e12 * (mask_npu - 1)                 # (B,1,S,S)
    # logits = logits + attention_mask.half()                  # 按旧脚本用 float16 加

    # weight = torch.softmax(logits.float(), dim=-1).half()    # 先 float32 稳数值，再降回 float16

    # attn = torch.bmm(weight.reshape(B * N, S, S), v2)        # (B*N,S,D)
    # attn = attn.reshape(B, N, S, D)

    # # ---------- 4. 落盘结果 ----------
    # attn.cpu().numpy().tofile('attn.bin')
   
    # print("Generate Data Finished!")

    B = int(list_data[0])
    H = int(list_data[1])
    N = int(list_data[2])
    M = int(list_data[3])
    K = int(list_data[4])
    D = int(list_data[5])
    scale = 1.0
    # # scale = 1.0
    is_float16 = int(list_data[6])
    if is_float16:
        dtype = np.float16
    else:
        dtype = np.bfloat16
    device = torch.device('npu')

    #######################################################生成数据+小算子拼接###############################################################
    # query = np.random.uniform(-1, 1, (B, H, N, M, D)).astype(dtype)
    # key = np.random.uniform(-1, 1, (B, H, N, K, D)).astype(dtype)
    # value = np.random.uniform(-1, 1, (B, H, N, K, D)).astype(dtype)
    # key1 = np.random.uniform(-1, 1, (B, H, K, M, D)).astype(dtype)
    # value1 = np.random.uniform(-1, 1, (B, H, K, M, D)).astype(dtype)
    # mask = np.concatenate((np.ones((B, H, N, M, K - K//2)).astype(np.float32), np.zeros((B, H, N, M, K//2)).astype(np.float32)), axis=-1)

    # evo_mask = 1 - mask.astype(np.uint8)
    # query.tofile(os.path.join(os.path.dirname(__file__), "query.bin"))
    # key.tofile(os.path.join(os.path.dirname(__file__), "key.bin"))
    # value.tofile(os.path.join(os.path.dirname(__file__), 'value.bin'))
    # key1.tofile(os.path.join(os.path.dirname(__file__), 'key1.bin'))
    # value1.tofile(os.path.join(os.path.dirname(__file__), 'value1.bin'))
    # evo_mask.tofile(os.path.join(os.path.dirname(__file__), 'atten_mask.bin'))
    # device = torch.device('npu')

    # q = torch.from_numpy(query).to(device)      # float16 -> NPU
    # k = torch.from_numpy(key).to(device)
    # v = torch.from_numpy(value).to(device)
    # mask_npu = torch.from_numpy(mask).to(device)  # float32

    # logits = torch.einsum('bhikc,bhijc->bhikj', q, k)*scale

    # attention_mask = 1e12 * (mask_npu - 1)                 # (B,1,S,S)
    # logits = logits + attention_mask.half()                  # 按旧脚本用 float16 加

    # weight, x_max, x_sum = tsoftmax(logits.float())

    # attn = torch.einsum('bhikj,bhijc->bhikc', weight, v)

    # attn.cpu().numpy().tofile('attn.bin')
    #######################################################################################################################################


    query = np.fromfile(os.path.join(os.path.dirname(__file__), "query.bin"), dtype=np.float16, ).reshape(B, H, N, M, D)
    key = np.fromfile(os.path.join(os.path.dirname(__file__), "key.bin"), dtype=np.float16).reshape(B, H, N, K, D)
    value = np.fromfile(os.path.join(os.path.dirname(__file__), "value.bin"), dtype=np.float16).reshape(B, H, N, K, D)
    key1 = np.fromfile(os.path.join(os.path.dirname(__file__), "key1.bin"), dtype=np.float16).reshape(B, H, K, M, D)
    value1 = np.fromfile(os.path.join(os.path.dirname(__file__), "value1.bin"), dtype=np.float16).reshape(B, H, K, M, D)
    
    mask = 1 - np.fromfile(os.path.join(os.path.dirname(__file__), "atten_mask.bin"), dtype=np.uint8).reshape(B, H, N, M, K)
    mask = mask.astype(np.float32)
    q = torch.from_numpy(query).to(device)      # float16 -> NPU
    k = torch.from_numpy(key).to(device)
    v = torch.from_numpy(value).to(device)
    k1 = torch.from_numpy(key1).to(device)
    v1 = torch.from_numpy(value1).to(device)
    mask_npu = torch.from_numpy(mask).to(device)  # float32
    logits = torch.einsum('bhikc,bhjkc->bhikj', q, k1)#  torch.einsum('bhikc,bhijc->bhikj', q, k)# + 
    attention_mask = 1e9 * (mask_npu - 1)                 # (B,1,S,S)
    logits = logits + attention_mask.half()                  # 按旧脚本用 float16 加
    weight, x_max, x_sum = tsoftmax(logits.float())
    attn = torch.einsum('bhikj,bhijc->bhikc', weight, v)# + torch.einsum('bhikj,bhijc->bhikc', weight, v1)
    attn.cpu().numpy().tofile('attn.bin')

else:
    raise RuntimeError(f"Invalid case name:", case_name)