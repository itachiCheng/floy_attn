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
 * \file fused_floyd_attention_s1s2_bn2gs1.h
 * \brief
 */

#ifndef FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H
#define FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H

#include "util.h"
#include "dropmask.h"
#include "fused_floyd_attention_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "pse.h"

using matmul::MatmulType;

struct SplitExtraInfo {
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopCount;
    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t taskId;
    int8_t taskIdMod2;
    int8_t multiCoreInnerIdxMod2;
    int8_t needNz2Nd;
    bool lastNotPair;
    int32_t s1RealSize;
    int32_t s2RealSize;
    int32_t s2AlignedSize;
    int32_t vec1S1BaseSize;
    int32_t vec1S1RealSize;
    int32_t vec2S1BaseSize;
    int32_t vec2S1RealSize;
    int32_t realSplitN;
    int32_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
    int64_t pseS2ComputeSize;
    int64_t s1SizeDelta;
    int64_t s1SizeAcc;
    int64_t s2SizeAcc;
    int64_t attenB1SSOffset;
    int64_t attenMaskS2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t softmaxMaxOffset;
};

constexpr int64_t GM_DOUBLE_BUFFER = 2;
constexpr int64_t INVALID_OFFSET = INT64_MIN;
constexpr int64_t SPLIT_S2_SIZE_LIMIT = 1024;
constexpr AscendC::SoftmaxConfig SOFTMAX_DEFAULT_CFG = {false};

__aicore__ const constexpr MatmulConfig &GetMmCfg(bool enableL1Reuse)
{
    if (enableL1Reuse) {
        return CFG_IBSHARE_EXCEED;
    } else {
        return CFG_EXCEED;
    }
}

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, CubeFormat bmm1Format = CubeFormat::ND,
          bool enableL1Reuse = false>
class FusedFloydAttentionS1s2Bn2gs1 {
public:
    __aicore__ inline FusedFloydAttentionS1s2Bn2gs1(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key0, __gm__ uint8_t *key1,
                                __gm__ uint8_t *value0, __gm__ uint8_t *value1,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FusedFloydAttentionGeneralTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, enableL1Reuse>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, GetMmCfg(enableL1Reuse)> bmm1;

    using c1NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    matmul::Matmul<a1Type, b1Type, c1NzType, bias1Type, GetMmCfg(enableL1Reuse)> bmm1Nz;

    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, enableL1Reuse>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using modeTypemm2 = typename AscendC::Conditional<
          (IsSameType<T, INPUT_T>::value == false && layOutType == LayOutTypeEnum::LAYOUT_TND),
          matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, GetMmCfg(enableL1Reuse)>,
          matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, GetMmCfg(enableL1Reuse)>>::type;
    modeTypemm2 bmm2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
                                     __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result(SplitExtraInfo &extraInfo);
    __aicore__ inline void WaitBmm2Result();
    __aicore__ inline void IterateBmm2(SplitExtraInfo &extraInfo);
    __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx, bool lastNotPair);
    __aicore__ inline void SetTiling(const FusedFloydAttentionGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void GetBNIdx(const int64_t &multiCoreInnerOffset, const int64_t &multiCoreInnerLimit);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void IterateBmm1(SplitExtraInfo &extraInfo,
                                       matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void Bmm1SetTensorA(SplitExtraInfo &extraInfo,
                                          matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void SetBmm1TensorB(SplitExtraInfo &extraInfo,
                                          matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
    __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset,
                                           bool secondTime = false);
    __aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre, int64_t s1Offset,
                                                   SplitExtraInfo &extraInfo);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void GetBmm1Result(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                            LocalTensor<uint8_t> &maskUb, const uint8_t maskType, event_t vWaitMte2);

    __aicore__ inline void SoftMaxCompute(SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void SoftMaxCheckResCompress(SplitExtraInfo &extraInfo, int64_t vec1S1realSplitN);
    __aicore__ inline void InvalidLineSplitS2Process(SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor,
                                                     LocalTensor<T> &maxUb, int64_t loopIdx);
    __aicore__ inline void ProcessVec2(SplitExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultMul(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void Bmm2DataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize);
    __aicore__ inline void SoftmaxDataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx);

    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void GetS2LoopRange(bool useNext, bool lastNotPair);

    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    int64_t dSizeAlign16;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1OuterSize;

    // sparse 用参数
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t nextS2EndIdx;

    // BNG 外循环
    int64_t bngStartIdx;
    int64_t bngEndIdx;

    // s2方向的尾块，包含N:1配比
    int64_t lastS2RealSize = INVALID_OFFSET;
    int64_t bmm2LastS2RealSize = INVALID_OFFSET;

    // L1Reuse场景vector核是否是奇数核
    int64_t l1ReuseBlockMod2 = 0;

    int64_t qCoreOffset;

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> maskTBufPong;
    TBuf<> pseTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> softmaxMaxBuf;
    TBuf<> commonTBuf; // common的复用空间

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<half> pseAlibiGm;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t n2GS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t bN2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2GD;
    int64_t gS2;

    // s2base*N之后的长度
    int64_t s2BaseNratioSize;
    int64_t s1BaseS2;

    int64_t s2BaseN2D;
    int64_t s2BaseBN2D;
    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s2BaseNratioD;
    int64_t s2BaseNratioN2D;
    int64_t s2BaseNratioBN2D;
    int64_t bN2G;
    int64_t n2GS2;
    int64_t s2SizeSum;

    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;
    // 当splitN大于16时，需要修改softMaxCheckRes数据类型
    uint16_t softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

    int32_t blockIdx;
    const FusedFloydAttentionGeneralTilingData *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

    TPipe *pipe;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> pseGm;
    __gm__ uint8_t *pseSlope;
    GM_ADDR prefixNAddr;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<uint8_t> attenMaskGmInt;

    bool dropMaskUnAligned;
    int64_t attenMaskOffsetPre = 0;
    PseInfo pseInfo = {0};
    DropMaskInfo dropMaskInfo = {0};
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key0, __gm__ uint8_t *key1,
                                                   __gm__ uint8_t *value0, __gm__ uint8_t *value1,
                                                   __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                                   __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                                   const FusedFloydAttentionGeneralTilingData *__restrict tiling,
                                                   TPipe *tPipe)
{

}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::Process()
{
    
};


#endif // FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H
