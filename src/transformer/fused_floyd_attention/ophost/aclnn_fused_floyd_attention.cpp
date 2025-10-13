/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_floyd_attention.h"
#include "fused_floyd_attention.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int64_t PAD_BASIC_BLOCK = 16;
static const int64_t PAD_LOWER_BOUND_196 = 196;
static const int64_t PAD_ALIGN_128 = 128;
static const int64_t PAD_ALIGN_SPL_SHAPE = 448;
static const int64_t MAX_STRIDE_S1 = 65535;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const int64_t HEAD_DIM_MAX = 512;
static const int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t PSE_INNER_MUL_ADD = 2;
static const int64_t PSE_INNER_MUL_ADD_SQRT = 3;
static const int64_t HEAD_DIM_72 = 72;
static const int64_t HEAD_DIM_88 = 88;
static const int64_t SEQ_LEN_1024 = 1024;
static const int64_t TND_UNPAD_MAX_S2 = 1024;
static const int64_t TND_UNPAD_MAX_S1_SUM = 160 * 1024;
static const int64_t TND_UNPAD_MAX_DDIM = 96;

struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
};

enum class InputLayout {
    BSND,
    SBH,
    BNSD,
    BSH,
    TND
};

struct FaShapeInfo {
    AxesInfo axes;

    InputLayout inputLayout;
    string l0InputLayoutStr;

    uint64_t dimNum = 0;
    uint64_t padNum = 0;

    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyValueShape;
    // FIXED ADD
    FVector<int64_t, DIM_NUM_4> reshapedKey1ValueShape;

    FVector<int64_t, DIM_NUM_4> reshapedAttenMaskShape;
    bool needPad = false;
    bool needTranspose = false;
    bool needReshape = false;
};

void AnalysisAxisForBnsd(const Shape &qShape, const Shape &kShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BNSD;
    shapeInfo.l0InputLayoutStr = "BNSD";

    // NMD NKD KMD
    shapeInfo.axes.b = qShape[0]*qShape[1];
    shapeInfo.axes.n2 = kShape[2];  // N
    shapeInfo.axes.s1 = qShape[3];  // M
    shapeInfo.axes.s2 = kShape[3];  // K
    shapeInfo.axes.d = qShape[4];
}

aclnnStatus AnalysisAxis(const aclTensor *query, const aclTensor *key_0, const aclTensor *key_1,
                         FaShapeInfo &shapeInfo)
{
    Shape kShape = key_0->GetViewShape();
    Shape k1Shape = key_1->GetViewShape();
    Shape qShape = query->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    std::string inputLayoutStr = "BNSD";

    if (inputLayoutStr == "BNSD") {
        AnalysisAxisForBnsd(qShape, kShape, shapeInfo);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayoutStr, shapeInfo.dimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}


void SetShapeInfoForBnsd(int64_t alignedH1Size, FaShapeInfo &shapeInfo)
{
    if (shapeInfo.inputLayout == InputLayout::BNSD) {
        shapeInfo.needReshape = true;
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.s1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyValueShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.s2, shapeInfo.axes.d});
        // FIXED
        shapeInfo.reshapedKey1ValueShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.s1, shapeInfo.axes.d});

        shapeInfo.reshapedAttenMaskShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.s1, shapeInfo.axes.s2});
    }
}


void SetShapeInfoForSbh(int64_t alignedH1Size, FaShapeInfo &shapeInfo)
{
    if (shapeInfo.axes.b * alignedH1Size > MAX_STRIDE_S1) {
        shapeInfo.needTranspose = true;
        shapeInfo.needReshape = true;
        shapeInfo.l0InputLayoutStr = "BNSD";

        // S,B,N,D -> B,N,S,D
        shapeInfo.perm_in.assign({1, 2, 0, 3});
        // B,N,S,D -> S,B,N,D
        shapeInfo.perm_out.assign({2, 0, 1, 3});
    }
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
    }

    if (shapeInfo.needReshape) {
        if (!shapeInfo.needTranspose) {
            shapeInfo.l0InputLayoutStr = "SBH";
        }
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.s1, shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyValueShape.assign(
            {shapeInfo.axes.s2, shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.d});
    }
}

static int64_t GetSumIntArrayMaxValue(const aclIntArray *intArrayValue)
{
    // 获取targetLengthsList中的最大值
    int64_t maxLength = 0;
    int64_t tmpMaxLength = 0;
    if (intArrayValue->Size() == 1) {
        maxLength = static_cast<int64_t>((*intArrayValue)[0]);
        return maxLength;
    }
    maxLength = static_cast<int64_t>((*intArrayValue)[0]);
    for (size_t i = 1; i < intArrayValue->Size(); ++i) {
        tmpMaxLength = static_cast<int64_t>((*intArrayValue)[i]) - static_cast<int64_t>((*intArrayValue)[i - 1]);
        if (tmpMaxLength > maxLength) {
            maxLength = tmpMaxLength;
        }
    }
    return maxLength;
}


aclnnStatus InputDtypeCheck(const aclTensor *query, const aclTensor *key_0, const aclTensor *value_0)
{
    auto vDtype = value_0->GetDataType();
    auto kDtype = key_0->GetDataType();
    auto qDtype = query->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], key[%s], value[%s] are not equal.",
                op::ToString(DataType(qDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AnalysisInput(const aclTensor *query, const aclTensor *key_0, const aclTensor *key_1,
                          FaShapeInfo &shapeInfo)
{
    CHECK_RET(AnalysisAxis(query, key_0, key_1, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head dim must <= 512, but got %ld", shapeInfo.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.d == 0) {
        return ACLNN_SUCCESS;
    }

    if (shapeInfo.inputLayout != InputLayout::TND &&
        (shapeInfo.axes.b == 0 || shapeInfo.axes.s1 == 0 || shapeInfo.axes.s2 == 0)) {
        return ACLNN_SUCCESS;
    }

    int64_t alignDim = (shapeInfo.axes.d < PAD_LOWER_BOUND_196 || shapeInfo.axes.d == PAD_ALIGN_SPL_SHAPE) ?
                           PAD_BASIC_BLOCK :
                           PAD_ALIGN_128;
    if (shapeInfo.axes.d % alignDim != 0) {
        shapeInfo.needPad = true;
        shapeInfo.padNum = (shapeInfo.axes.d + alignDim - 1) / alignDim * alignDim - shapeInfo.axes.d;
    }

    int64_t alignedH1Size = shapeInfo.axes.n1 * (shapeInfo.axes.d + shapeInfo.padNum);

    if (shapeInfo.inputLayout == InputLayout::BNSD) {
        SetShapeInfoForBnsd(alignedH1Size, shapeInfo);
    }
    OP_LOGD("Analysis input success. The analysis result: [needReshape]: %d, [needPad]: %d, [padNum]: %lu,"
            "[needTranspose]: %d.",
            shapeInfo.needReshape, shapeInfo.needPad, shapeInfo.padNum, shapeInfo.needTranspose);
    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

aclnnStatus Contiguous(const aclTensor *&query, const aclTensor *&key_0, const aclTensor *&value_0, const aclTensor *&key_1, const aclTensor *&value_1,
                       const aclTensor *&attenMaskOptional,
                       aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key_0 = l0op::Contiguous(key_0, executor);
    CHECK_RET(key_0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key_1 = l0op::Contiguous(key_1, executor);
    CHECK_RET(key_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value_0 = l0op::Contiguous(value_0, executor);
    CHECK_RET(value_0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value_1 = l0op::Contiguous(value_1, executor);
    CHECK_RET(value_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus PreprocessQKV(const aclTensor *&query, const aclTensor *&key_0, const aclTensor *&value_0, const aclTensor *&key_1, const aclTensor *&value_1, const aclTensor *&attenMaskOptional,
                          const struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needReshape) {
        query = l0op::Reshape(
            query, executor->AllocIntArray(shapeInfo.reshapedQueryShape.data(), shapeInfo.reshapedQueryShape.size()),
            executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key_0 = l0op::Reshape(
            key_0,
            executor->AllocIntArray(shapeInfo.reshapedKeyValueShape.data(), shapeInfo.reshapedKeyValueShape.size()),
            executor);
        CHECK_RET(key_0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        value_0 = l0op::Reshape(
            value_0,
            executor->AllocIntArray(shapeInfo.reshapedKeyValueShape.data(), shapeInfo.reshapedKeyValueShape.size()),
            executor);
        CHECK_RET(value_0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        key_1 = l0op::Reshape(
            key_1,
            executor->AllocIntArray(shapeInfo.reshapedKey1ValueShape.data(), shapeInfo.reshapedKey1ValueShape.size()),
            executor);
        CHECK_RET(key_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        value_1 = l0op::Reshape(
            value_1,
            executor->AllocIntArray(shapeInfo.reshapedKey1ValueShape.data(), shapeInfo.reshapedKey1ValueShape.size()),
            executor);
        CHECK_RET(value_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        attenMaskOptional = l0op::Reshape(
            attenMaskOptional,
            executor->AllocIntArray(shapeInfo.reshapedAttenMaskShape.data(), shapeInfo.reshapedAttenMaskShape.size()),
            executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Postprocess(const aclTensor *&l0AttentionOutOut, const aclTensor *attentionOutOut,
                        struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    auto attentionOutOutShape = ToShapeVector(attentionOutOut->GetViewShape());
    l0AttentionOutOut =
        l0op::Reshape(l0AttentionOutOut,
                        executor->AllocIntArray(attentionOutOutShape.data(), attentionOutOutShape.size()), executor);
    CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFaParam(const aclTensor *query, const aclTensor *key_0, const aclTensor *value_0, const aclTensor *key_1, const aclTensor *value_1,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *attentionOutOut,
    const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(key_0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value_0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(key_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(softmaxMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(softmaxSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedFloydAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key_0, const aclTensor *value_0, const aclTensor *key_1, const aclTensor *value_1, const aclTensor *attenMaskOptional,
    double scaleValueOptional, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key_0, value_0, key_1, value_1, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnFusedFloydAttention,
                   DFX_IN(query, key_0, value_0, key_1, value_1,
                          attenMaskOptional, scaleValueOptional),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }


    CHECK_RET(InputDtypeCheck(query, key_0, value_0) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key_0, key_1, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key_0, value_0, key_1, value_1, attenMaskOptional,
                         l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key_0, value_0, key_1, value_1, attenMaskOptional, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0FusedFloydAttentionOuts = l0op::FusedFloydAttention(
        query, key_0, value_0, key_1, value_1, attenMaskOptional,
        scaleValueOptional, l0Executor);


    CHECK_RET(l0FusedFloydAttentionOuts[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0FusedFloydAttentionOuts[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0FusedFloydAttentionOuts[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto l0SoftmaxMaxOut = l0FusedFloydAttentionOuts[0];
    auto l0SoftmaxSumOut = l0FusedFloydAttentionOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FusedFloydAttentionOuts[2];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(Postprocess(l0SoftmaxMaxOut, softmaxMaxOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(Postprocess(l0SoftmaxSumOut, softmaxSumOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedFloydAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedFloydAttention);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif
