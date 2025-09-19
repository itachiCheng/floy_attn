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
    this->InitInput(query, key, value, attenMask, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe); // gm设置

}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key0, __gm__ uint8_t *key1,
                                                   __gm__ uint8_t *value0, __gm__ uint8_t *value1,
                                                   __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                                   __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                                   const FusedFloydAttentionGeneralTilingData *__restrict tiling,
                                                   TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->SetTiling(tiling);

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key0);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value0);
    // this->pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    // this->pseSlope = pse;
    // this->prefixNAddr = prefix;
    // this->dropMaskUnAligned = this->tilingData->inputParams.needDropMaskOp == 1;
    // if (this->dropMaskUnAligned) {
    //     this->dropMaskGm.SetGlobalBuffer(workspace);
    //     if constexpr (hasDrop == true) {
    //         workspace += CeilDiv(this->tilingData->dropmaskParams.shapeTotalSize, 512) * 512;
    //     }
    // } else {
    //     this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    // }
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    // 补齐到512， 统一按T处理
    int64_t mm1ResultSize = s1BaseSize * s2BaseSize;
    int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T);
    int64_t mm2ResultSize = s1BaseSize * dSizeAlign16;
    int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
    int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
    int64_t vector1OffsetPing = 0;
    int64_t vector1OffsetPong = mmNRatioOffset;

    // NZND场景，stage1Result不与bmm1Result共用空间，需要占用1倍mmNRatioOffset空间
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
            vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
            vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
            bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;
        } else {
            if (this->tilingData->inputParams.s2Size % 64 != 0) {
                vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
                vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
                bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;
            }
        }
    }

    // FP32场景，stage1Result不与bmm1Result共用空间，需要占用2倍mmNRatioOffset空间
    if constexpr (IsSameType<INPUT_T, float>::value) {
        vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
        vector1OffsetPong = vector1OffsetPing + mmNRatioOffset;
        bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 2;
    }

    int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * GM_DOUBLE_BUFFER;
    if (dSizeAlign16 > 64) {
        totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;
    }

    // int64_t pseInnerAlibiSize = this->tilingData->coreParams.pseAlibiBaseS1 *
    //                             this->tilingData->coreParams.pseAlibiBaseS2 * sizeof(half);
    // int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;
    // totalOffset += pseAlibiOffset;

    // bmm1Result，占用2倍mmNRatioOffset空间
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));

    // stage1Result，不占用/占用1倍/占用2倍mmNRatioOffset空间
    this->stage1Res[0].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + vector1OffsetPing));
    this->stage1Res[1].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + vector1OffsetPong));

    // bmm2Result，占用2倍mmOffset空间
    this->mm2Res[0].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
    this->mm2Res[1].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));

    // uint64_t pseAlibiAddr = this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + 2 * mm2Offset;

    // vec2阶段，占用2倍mmOffset空间，仅在D轴大于64的情况下出现
    if (dSizeAlign16 > 64) {
        this->vec2Res[0].SetGlobalBuffer(
            (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2));
        this->vec2Res[1].SetGlobalBuffer(
            (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 3));
        // pseAlibiAddr = this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + 4 * mm2Offset;
    }
    // this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr));
    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::InitBuffer()
{
    uint64_t stage1Size = 8 * 1024;
    uint64_t stage1AttenSize = 9 * 1024;
    uint64_t stage1PongSize = 35 * 1024;
    uint64_t stage2Size = 64 * 128;
    uint64_t maskTBufPongSize = 16 * 1024;

    // 可选输入的buffer空间，保持和stage1处理的size一致
    this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize); // 可以给attenmask 9k
    this->pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize); // 可以给dropoutmask 16k
    // this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
    this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumBuf[0], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxSumBuf[1], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxMaxBuf, s1BaseSize * blockBytes);    // 4k
    this->pipe->InitBuffer(this->softmaxExpBuf[0], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxExpBuf[1], s1BaseSize * blockBytes); // 4k
    if constexpr (bmm1Format == CubeFormat::NZ) {
        this->pipe->InitBuffer(this->stage1PongBuf, stage1PongSize); // i.a 35k
    } else {
        this->pipe->InitBuffer(this->stage1PongBuf, stage1Size * sizeof(T)); // i.a 32k
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::ComputeConstexpr()
{
    // 计算轴的乘积
    if constexpr (enableL1Reuse) {
        this->s1OuterSize = this->tilingData->coreParams.s1OuterSize;
    }
    this->s1D = this->tilingData->inputParams.s1Size * dSize;
    this->s2D = this->tilingData->inputParams.s2Size * dSize;
    this->gD = this->tilingData->inputParams.gSize * dSize;
    this->n2D = this->tilingData->inputParams.n2Size * dSize;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;

    this->bN2D = this->tilingData->inputParams.bSize * n2D;
    this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
    this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->bN2GD = this->tilingData->inputParams.bSize * n2GD;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;

    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;
    this->s1BaseS2 = this->s1BaseSize * this->tilingData->inputParams.s2Size;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        this->s1BaseN2GD = this->s1BaseSize * this->n2GD;
        this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;
        this->mm1Ka = this->n2GD;
        this->mm1Kb = this->n2D;
        this->mm2Kb = this->n2D;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        this->bN2G = this->tilingData->inputParams.bSize * this->n2G;
        this->s1BaseBN2GD = s1BaseSize * this->tilingData->inputParams.bSize * this->n2GD;
        this->s2BaseBN2D = this->tilingData->inputParams.bSize * this->s2BaseN2D;
        this->s2BaseNratioBN2D = this->s2BaseBN2D * this->tilingData->coreParams.nRatio;
        this->mm1Ka = this->bN2GD;
        this->mm1Kb = this->bN2D;
        this->mm2Kb = this->bN2D;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        this->s1BaseD = this->s1BaseSize * this->dSize;
        this->s2BaseNratioD = this->s2BaseNratioSize * this->dSize;
        this->mm1Ka = this->dSize;
        this->mm1Kb = this->dSize;
        this->mm2Kb = this->dSize;
    }

    if (this->tilingData->inputParams.pseShapeType == pse1S2) {
        this->gS2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
        this->n2GS2 = this->tilingData->inputParams.n2Size * this->gS2;
    }
    // if constexpr (hasPse == true) {
    //     this->pseInfo.gSize = this->tilingData->inputParams.gSize;
    //     this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
    //     this->pseInfo.pseType = this->tilingData->inputParams.pseType;
    //     this->pseInfo.n2G = this->n2G;
    //     this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
    //     this->pseInfo.s1BaseSize = this->s1BaseSize;
    //     this->pseInfo.pseS1Size = this->tilingData->inputParams.pseS1Size;
    //     this->pseInfo.pseS2Size = this->tilingData->inputParams.pseS2Size;
    //     this->pseInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    //     this->pseInfo.pseEncodeType = (uint32_t)this->tilingData->inputParams.pseEncodeType;
    //     this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
    //     this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
    //     this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
    //     this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
    // }
    // if constexpr (hasDrop == true) {
    //     this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
    //     this->dropMaskInfo.n2G = this->n2G;
    //     this->dropMaskInfo.s1BaseSize = this->s1BaseSize;
    //     this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    // }
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::Process()
{
    
};


#endif // FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H
