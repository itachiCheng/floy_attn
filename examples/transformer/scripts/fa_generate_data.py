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

import numpy as np
import sys

case_name = sys.argv[1]
if case_name == 'test_flash_attention_score':
    query = np.random.uniform(-0.1, 0.1, (2048, 1, 128)).astype(np.float16)
    key = np.random.uniform(-0.1, 0.1, (2048, 1, 128)).astype(np.float16)
    value = np.random.uniform(-0.1, 0.1, (2048, 1, 128)).astype(np.float16)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')

elif case_name == 'test_fused_floyd_attention':
    B = 1
    H = 6
    N = 100
    M = 100
    K = 100
    D = 32
    query = np.random.uniform(-0.1, 0.1, (B, H, N, M, D)) # BHNMD
    key_0 = np.random.uniform(-0.1, 0.1, (B, H, N, K, D)) # BHNKD
    key_1 = np.random.uniform(-0.1, 0.1, (B, H, K, M, D)) # BHKMD
    value_0 = np.random.uniform(-0.1, 0.1, (B, H, N, K, D)) # BHNKD
    value_1 = np.random.uniform(-0.1, 0.1, (B, H, K, M, D)) # BHKMD
    query.tofile('query.bin')
    key_0.tofile('key_0.bin')
    key_1.tofile('key_1.bin')
    value_0.tofile('value_0.bin')
    value_1.tofile('value_1.bin')
    
else:
    raise RuntimeError(f"Invalid case name:", case_name)
