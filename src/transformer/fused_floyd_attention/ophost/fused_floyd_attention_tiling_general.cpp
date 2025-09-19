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
 * \file fused_floyd_attention_tiling_general.cpp
 * \brief
 */

#include <numeric>
#include "tiling/data_copy_transpose_tiling.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "fused_floyd_attention_tiling.h"
#include "fused_floyd_attention_tiling_common.h"

namespace optiling {
namespace FLOYD {
const uint32_t BYTE_BLOCK = 32;
const int64_t GM_ALIGN = 512;
const int64_t FRACTAL_NUM = 16L;
const int64_t PSE_DIM_NUM = 4L;
const int64_t S1_VEC2_BASE_SIZE_MAX = 512L;

const int64_t BYTE_BIT_NUM = 8UL;
const size_t PSE_INPUT_INDEX = 3UL;
const size_t DROP_MASK_INPUT_INDEX = 4UL;
const size_t ATTENTION_MASK_INPUT_INDEX = 6UL;
const size_t PREFIX_INPUT_INDEX = 7UL;
const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
const size_t ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
const size_t Q_START_IDX_INPUT_INDEX = 10UL;
const size_t KV_START_IDX_INPUT_INDEX = 11UL;
const size_t ATTEN_OUT_INDEX = 2UL;
const size_t ATTENTION_MASK_DIM_NUM_4 = 4UL;
const size_t ATTENTION_MASK_DIM_NUM_2 = 2UL;
const int64_t BMM_SOFTMAX_RATIO = 4L;
const int64_t MAX_AIV_NUM = 48L;
const int64_t DROP_MASK_ALIGN_UNIT = 256L; // input bits, and align to 32B in UB
const int64_t HIGH_PERF_BUFFER_NUM = 6L;
const int64_t HIGH_PERF_SUPPORT_S2_BASIC = 128L;
const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
const uint32_t PSE_ALIBI_S_SIZE = 1024;

constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
const int64_t ATTEN_MASK_S1_REV_INDEX = 2L;
const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;
const int64_t ATTEN_MASK_COMPRESS_PREFIX_LIMIT = 3072L;
const int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
const int64_t S2_REUSE_SIZE_512 = 512L;
const int64_t S2_REUSE_SIZE_1024 = 1024L;
const int64_t S1_REUSE_SIZE_3840 = 3840L;
const int64_t D_SPECIFIC_SIZE = 64L;
const int64_t BALANCE_LOAD_LIST_SIZE = 8L;
constexpr int64_t COF[BALANCE_LOAD_LIST_SIZE] = {256, 384, 512, 640, 768, 896, 960, 1024};
const int64_t BMM1_BASICBLOCK_M_128 = 128L;
const int64_t BMM1_BASICBLOCK_N_256 = 256L;
const int64_t BMM1_BASICBLOCK_N_128 = 128L;
const int64_t BMM1_BASICBLOCK_K_64 = 64L;
const int64_t BMM1_BASICBLOCK_K_128 = 128L;
const int64_t S2_NZTOND_SIZE_64 = 64L;
const int64_t SPACE_NUM_2 = 2L;
const int64_t SPACE_NUM_3 = 3L;
const int64_t SPACE_NUM_4 = 4L;
const int64_t BMM2_BASICBLOCK_M_64 = 64L;
const int64_t BMM2_BASICBLOCK_N_64 = 64L;
const int64_t BMM2_BASICBLOCK_K_256 = 256L;
const int64_t S2_SPECIFIC_SIZE_928 = 928L;
const int64_t S2_NZTOND_SIZE_128 = 128L;
const int64_t UB_BASIC_LIMIT_SIZE = 8 * 1024;
const int64_t SLOPE_BN_DIM_NUM = 2L;
const int64_t SLOPE_N_DIM_NUM = 1L;
const int64_t L1REUSE_D_Limit = 128L;
const int64_t L1REUSE_BNG_Limit = 10L;
const int64_t L1REUSE_S2_Limit_1024 = 1024;
const int64_t L1REUSE_S2_Limit_2048 = 2048L;
const int64_t L1REUSE_S2_LIMIT_256 = 256;
const int64_t L1REUSE_S2_LIMIT_4032 = 4032;
const int64_t AICAIV_RATIO_2 = 2L;
const int64_t L1REUSE_S2_LIMIT_512 = 512;
const int64_t L1REUSE_BNG_LIMIT_64 = 64;
const int64_t L1REUSE_BNG_LIMIT_4800 = 4800L;
const int64_t L1REUSE_D_LIMIT_144 = 144L;
const int64_t INVALID_ROW_SPARSE_RATIO = 6L;
const int64_t HEAD_DIM_MAX_VALUE = 512L;
const int64_t DATA_TYPE_FP16 = 2L;
const int64_t DATA_TYPE_FP32 = 4L;
enum LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
    ATTEN_1_1_1_T_T = 99,
};

enum PseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

enum PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE, // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};


enum PseEncodeType : uint8_t {
    PES_ENCODE_NONE = 0,
    PSE_ENCODE_ALIBI_S2_FULL = 0x11, // shape: (1024, S2)
};

template <typename T> static T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T> static T AlignDown(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T> static T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T> static T CeilDiv(const T n1, const T n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T> static T CalcTailSize(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

class TilingKey {
public:
    TilingKey() : splitS1(0), splitS2(0), splitD(0), dtype(0), layoutType(0), sparseType(0), reserved(0)
    {
    }

    void Reset()
    {
        splitS1 = 0;
        splitS2 = 0;
        splitD = 0;
        dtype = 0;
        layoutType = 0;
        sparseType = 0;
        reserved = 0;
    }

    uint32_t GetRawTilingKey() const
    {
        return *(reinterpret_cast<const uint32_t *>(this));
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << " splitS1: " << splitS1 << " splitS2: " << splitS2 << " splitD: " << splitD;
        return ss.str();
    }

    uint32_t splitS1    : 1;
    uint32_t splitS2    : 1;
    uint32_t splitD     : 1;
    uint32_t dtype      : 2;
    uint32_t layoutType : 2;
    uint32_t sparseType : 2;
    uint32_t reserved   : 23; // to fullfil 32 bit, if add new template bit then decrease this number
};

inline bool operator==(const TilingKey &left, const TilingKey &right)
{
    return left.GetRawTilingKey() == right.GetRawTilingKey();
}

using TemplateType = TilingKey;

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class FusedFloydAttentionTilingBase : public TilingBaseClass {
public:
    explicit FusedFloydAttentionTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~FusedFloydAttentionTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    void SetSparseTilingInfo(SparseEnum &sparseType);
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    virtual void GetBufferNum(BufferNum &bufferNum) const = 0;

    void Reset();

    virtual int64_t GetNRatio();

    virtual int64_t GetMinS1BasicBlock() const
    {
        return std::min(64L, alignedS1);
    }

    virtual bool IsTemplateMatched() const
    {
        return expectTemplate == actualTemplate;
    }

    ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    // bool Analyze5DimLayout(const gert::Shape &queryShape, const gert::Shape &key0Shape, const gert::Shape &key1Shape);
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &key0Shape, const gert::Shape &key1Shape);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock(const BufferNum &bufferNum);
    virtual void CalcDBasicBlock();
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const;
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock) const;
    virtual bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch = 1) = 0;
    bool IsBasicBlockInSoftMax(const ge::Shape &shape) const;
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm1);
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) = 0;
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                         matmul_tiling::MatmulApiTiling &bmm1, matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual void SetCoreParams();
    virtual void SetMultiBatchCoreParams();
    virtual void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    void SetDataCopyTransposeTiling();
    virtual void SetTensorSizeParams();
    uint32_t aivNum;
    uint32_t aicNum;
    int64_t apiMaxUBSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutType tilingKeyLayout;
    ImplMode implMode;
    CubeFormatEnum tilingKeyBmm1Format = CubeFormatEnum::ND;
    CubeInputSourceEnum tilingKeyBmm1Source = CubeInputSourceEnum::GM;
    CubeInputSourceEnum tilingKeyBmm2Source = CubeInputSourceEnum::GM;

    // FloyD
    int64_t BSize;
    int64_t HSize;
    int64_t NSize;
    int64_t MSize;
    int64_t KSize;
    int64_t DSize;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t preTokens;
    int64_t nextTokens;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t sparseMode;
    int64_t pseType;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t maxS1Val;
    int64_t minS1Val;
    int64_t accumS1;
    int64_t accumS1BlockNum;
    int64_t dropTotalSize;
    int64_t maxS2Val;
    int64_t minS2Val;
    int64_t accumS2;
    int64_t bandIndex;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    float keepProb;
    float scaleValue;
    uint8_t attenMaskCompressMode;
    uint8_t pseExistFlag;
    uint8_t attenMaskExistFlag;
    uint8_t dropMaskExistFlag;

    int64_t alignedN2;
    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t dBasicBlock;
    int64_t batchBasic;
    int64_t nRatio;

    int64_t minUsedUBSize;
    int64_t maxValidS2Len;

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;
    const int64_t *prefixNData;
    TemplateType expectTemplate;
    TemplateType actualTemplate;

    bool isSparseValidSizeAligned = false;
    bool hasPse = false;
    bool hasAttenMask = false;
    bool hasDropOut = false;
    FusedFloydAttentionGeneralTilingData tilingData;
};

int64_t FusedFloydAttentionTilingBase::GetNRatio()
{
    return BMM_SOFTMAX_RATIO;
}

ge::graphStatus FusedFloydAttentionTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FusedFloydAttentionCompileInfo *>(context_->GetCompileInfo());
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OPS_LOG_I(context_, "get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED)
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OPS_LOG_E_IF_NULL(context_, scaleValuePtr, return ge::GRAPH_FAILED)

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto key0Shape = context_->GetInputShape(1);
    auto key1Shape = context_->GetInputShape(2);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OPS_LOG_E_IF_NULL(context_, queryShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, queryDesc, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, key0Shape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, key1Shape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, attenOutShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, context_->GetRawTilingData(), return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, context_->GetRawTilingData()->GetData(), return ge::GRAPH_FAILED)
    OPS_ERR_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                                           context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
void FusedFloydAttentionTilingBase::SetSparseTilingInfo(SparseEnum &sparseType)
{
    auto &inputParams = tilingData.inputParams;
    inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
    inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
}

ge::graphStatus FusedFloydAttentionTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OPS_LOG_D_FULL(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OPS_ERR_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    alignedN2 = AlignUp(n2Size, FRACTAL_NUM);
    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);

    OPS_ERR_IF(alignedS1 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS1 %ld.", alignedS1),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(alignedS2 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS2 %ld.", alignedS2),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(alignedD <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedD %ld.", alignedD),
        return ge::GRAPH_FAILED);

    auto &inputParams = tilingData.inputParams;
    inputParams.set_BSize(BSize);
    inputParams.set_HSize(HSize);
    inputParams.set_NSize(NSize);
    inputParams.set_MSize(MSize);
    inputParams.set_KSize(KSize);

    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_pseType(static_cast<uint32_t>(pseType));
    OPS_LOG_D(context_, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], scaleValue[%f]",
    bSize, n2Size, gSize, s1Size, s2Size, dSize, scaleValue);
    std::cout << "input params: bn2gs1s2d" << bSize << ' ' << n2Size << std::endl;
    return ge::GRAPH_SUCCESS;
}

void FusedFloydAttentionTilingBase::Reset()
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
    apiMaxUBSize = 0;

    bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = inputDtypeBytes;

    tilingKeyDType = DtypeEnum::FLOAT16;
    tilingKeyLayout = LayoutType::LAYOUT_BNSD;
    tilingKeyBmm1Format = CubeFormatEnum::ND;
    tilingKeyBmm1Source = CubeInputSourceEnum::GM;

    bSize = 0LL;
    gSize = 0LL;
    dSize = 0LL;
    n1Size = 0LL;
    n2Size = 0LL;
    s1Size = 0LL;
    s2Size = 0LL;
    maxS1Val = 0LL;
    minS1Val = 0LL;
    accumS1 = 0LL;
    accumS2 = 0LL;
    bandIndex = 0LL;
    dropTotalSize = 0LL;
    maxS2Val = 0LL;
    minS2Val = 0LL;

    s1StrideSize = 0LL;
    s2StrideSize = 0LL;
    preTokens = std::numeric_limits<int32_t>::max();
    nextTokens = std::numeric_limits<int32_t>::max();
    sparseMode = static_cast<int64_t>(NO_MASK);
    pseType = PSE_OUTER_ADD_MUL_TYPE;
    pseAlibiBaseS1 = 0;
    pseAlibiBaseS2 = 0;
    qStartIdx = 0;
    kvStartIdx = 0;
    keepProb = 1.0f;
    scaleValue = 1.0f;
    pseExistFlag = 0;
    attenMaskCompressMode = NO_COMPRESS_MODE;
    attenMaskExistFlag = 0;
    dropMaskExistFlag = 0;
    isHighPercision = true;

    alignedN2 = 0LL;
    alignedS1 = 0LL;
    alignedS2 = 0LL;
    alignedD = 0LL;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();
    nRatio = GetNRatio();

    minUsedUBSize = 0LL;
    maxValidS2Len = 0LL;
    batchBasic = 1LL;

    opName = nullptr;
    inputLayout = nullptr;

    actualTemplate.Reset();
}

bool FusedFloydAttentionTilingBase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            calcTypeSize = ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OPS_LOG_D(context_, "Get high precision flag: %d.", isHighPercision);
    return true;
}

bool FusedFloydAttentionTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx);

    scaleValue = *scaleValuePtr;

    implMode = ImplMode::AA_HIGH_PRECISION;
    OPS_LOG_D(context_, "attrs: scale_value[%f].",
              scaleValue);
    // isHighPercision = true; // use default value
    return true;
}

bool FusedFloydAttentionTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &key0Shape = context_->GetInputShape(1)->GetStorageShape();
    auto &key1Shape = context_->GetInputShape(2)->GetStorageShape();

    // size_t layoutLen = strlen(inputLayout);
    // OPS_LOG_D(context_, "Get input_layout [%s].", inputLayout);
    OPS_ERR_IF(queryShape.GetDimNum() != 4 || key0Shape.GetDimNum() != 4 || key1Shape.GetDimNum() != 4,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Invalid layout, not 5 dim"), return false);
    OPS_ERR_IF(!Analyze4DimLayout(queryShape, key0Shape, key1Shape),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Get unsupported layout: 5 dim"), return false);
    // OPS_ERR_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero."), return false);
    // OPS_ERR_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero."), return false);
    // OPS_ERR_IF(dSize > HEAD_DIM_MAX_VALUE || dSize <= 0L,
    //            OPS_REPORT_VECTOR_INNER_ERR(opName, "dSize is not in range:(0, 512]."), return false);
    // OPS_ERR_IF(n1Size % n2Size != 0,
    //            OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld].", n1Size, n2Size),
    //            return false);
    return true;
}

bool FusedFloydAttentionTilingBase::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &key0Shape, const gert::Shape &key1Shape)
{
    // TODO
    // BSize = queryShape.GetDim(0);
    // HSize = queryShape.GetDim(1);
    // NSize = queryShape.GetDim(2);
    // KSize = key0Shape.GetDim(3);
    // MSize = key1Shape.GetDim(3);
    // DSize = queryShape.GetDim(4);

    bSize = queryShape.GetDim(0);
    
    n2Size = key0Shape.GetDim(1);
    gSize = queryShape.GetDim(1) / n2Size;
    s1Size = queryShape.GetDim(2); // 2: S1 idx
    std::cout << "s1Size" << s1Size << std::endl;
    s2Size = key0Shape.GetDim(2); // 2: S2 idx
    dSize = queryShape.GetDim(3); // 3: D idx
    s1StrideSize = dSize;
    s2StrideSize = dSize;
    tilingData.inputParams.set_layoutType(LAYOUT_BNSD);
    tilingKeyLayout = LayoutType::LAYOUT_BNSD;
    return true;

}

bool FusedFloydAttentionTilingBase::AnalyzeOptionalInput()
{
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    tilingData.inputParams.set_attenMaskDataType(1);
    attenMaskExistFlag = 1;
    AttenMaskShapeType attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
    tilingData.inputParams.set_attenMaskShapeType(attenMaskShapeType);
    OPS_LOG_D(context_, "attenMaskExistFlag: %d.", attenMaskExistFlag);
    return true;
}

ge::graphStatus FusedFloydAttentionTilingBase::DoOpTiling()
{
    auto &inputParams = tilingData.inputParams;
    OPS_LOG_D(context_, "[%s]try template[%s]", templateName, expectTemplate.ToString().c_str());
    if (!MatchTemplate()) {
        OPS_LOG_I(context_,
                  "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], "
                  "keepProb[%f]",
                  templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(), inputParams.get_n2Size(),
                  inputParams.get_gSize(), inputParams.get_s1Size(), inputParams.get_s2Size(), inputParams.get_dSize(),
                  inputParams.get_keepProb());
        return ge::GRAPH_PARAM_INVALID;
    }

    SparseEnum sparseType = SparseEnum::ALL;
    SetSparseTilingInfo(sparseType);
    inputParams.set_implMode(implMode);
    if (!isSparseValidSizeAligned) {
        s1SparseValidSize = preTokens;
        s2SparseValidSize = nextTokens;
    }
    SetCoreParams();
    SetMultiCoreParams();
    SetTensorSizeParams();
    return ge::GRAPH_SUCCESS;
}

bool FusedFloydAttentionTilingBase::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock(bufferNum);

    if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
        OPS_LOG_D(context_,
                  "[%s]can't find proper S1S2 basic block for shape: S1[%ld] S2[%ld], D[%ld], minS1BasicBlock[%ld], "
                  "dtype[%s], high precision[%d]",
                  templateName, s1Size, s2Size, dSize, GetMinS1BasicBlock(),
                  ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), isHighPercision);
        return false;
    }

    CalcDBasicBlock();
    actualTemplate.splitS1 = s1BasicBlock < alignedS1 ? 1 : 0;
    actualTemplate.splitS2 = s2BasicBlock < alignedS2 ? 1 : 0;
    actualTemplate.splitD = dBasicBlock < alignedD ? 1 : 0;

    if (IsTemplateMatched()) {
        (void)CalcUBSize(s1BasicBlock, s2BasicBlock);
        OPS_LOG_D(context_, "[%s]final basic block: [%ld, %ld, %ld], match template[%s].", templateName, s1BasicBlock,
                  s2BasicBlock, dBasicBlock, actualTemplate.ToString().c_str());
        return true;
    }

    return false;
}

void FusedFloydAttentionTilingBase::CalcS1S2BasicBlock(const BufferNum &bufferNum)
{
    // calc s1 s2 first, we set d basic block as s2 now
    const int64_t actualD = expectTemplate.splitD == 0 ? alignedD : FRACTAL_NUM; // if split d we use min s2 16
    int64_t maxS1BasicBlock = CalcMaxS1BasicBlockSize(actualD, bufferNum);
    maxS1BasicBlock = std::min(maxS1BasicBlock, alignedS1);
    if (maxS1BasicBlock == 0) {
        return;
    }

    for (int64_t tmpS1BasicBlock = std::min(GetMinS1BasicBlock(), maxS1BasicBlock); tmpS1BasicBlock <= maxS1BasicBlock;
         tmpS1BasicBlock += FRACTAL_NUM) {
        int64_t tmpS2BasicBlock = CalcMaxS2BasicBlockSize(bufferNum, tmpS1BasicBlock);
        tmpS2BasicBlock = std::min(tmpS2BasicBlock, alignedS2);
        for (; tmpS2BasicBlock >= FRACTAL_NUM; tmpS2BasicBlock -= FRACTAL_NUM) {
            // drop mask bug workaround
            if (dropMaskExistFlag == 1 &&
                (tmpS2BasicBlock <= BYTE_BLOCK || CalcTailSize(alignedS2, tmpS2BasicBlock) <= BYTE_BLOCK)) {
                continue;
            }

            int64_t tmpDBasicBlock = expectTemplate.splitD == 1 ? std::min(tmpS2BasicBlock, alignedD) : alignedD;
            OPS_LOG_D(context_, "[%s]try basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
            if (CalcUBSize(tmpS1BasicBlock, tmpS2BasicBlock, 1) &&
                SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock)) {
                break;
            }
        }

        // check whether is valid, if tmpS1BasicBlock is too big, then there is no proper tmpS2BasicBlock
        if (tmpS2BasicBlock < FRACTAL_NUM) {
            break;
        }

        OPS_LOG_D(context_, "[%s]get candidate basic block: [%ld, %ld]", templateName, tmpS1BasicBlock,
                  tmpS2BasicBlock);
        if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
            s1BasicBlock = tmpS1BasicBlock;
            s2BasicBlock = tmpS2BasicBlock;
        } else if (s2BasicBlock == tmpS2BasicBlock && s1BasicBlock < tmpS1BasicBlock) {
            s1BasicBlock = tmpS1BasicBlock;
        } else {
            break;
        }
    }
}

void FusedFloydAttentionTilingBase::CalcDBasicBlock()
{
    return;
}

int64_t FusedFloydAttentionTilingBase::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / inputDtypeBytes;
    int64_t maxS1BasicBlock =
        aicoreParams_.ubSize / inputDtypeBytes /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t FusedFloydAttentionTilingBase::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
                                                               int64_t tmpS1BasicBlock) const
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock;
    if (expectTemplate.splitD == 0) {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
                           tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * inputDtypeBytes) /
                          (tmpS1BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes);
    } else {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK) /
                          (tmpS1BasicBlock * (bufferNum.bufferS1DNum + bufferNum.bufferS1S2Num) * inputDtypeBytes);
    }
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool FusedFloydAttentionTilingBase::IsBasicBlockInSoftMax(const ge::Shape &shape) const
{
    // 2 axes at least
    if (shape.GetDimNum() < 2) {
        return false;
    }

    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }

    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }

    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void FusedFloydAttentionTilingBase::SetCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_s1BaseSize(s1BasicBlock);
    coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
    coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    coreParams.set_s2BaseSize(s2BasicBlock);
    coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
    coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
    if (expectTemplate.splitS2 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s2OuterSize());
        coreParams.set_s2OuterSize(CeilDivision(coreParams.get_s2OuterSize(), nRatio));
    } else if (expectTemplate.splitS1 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s1OuterSize());
    } else {
        nRatio = 1;
    }
    coreParams.set_nRatio(nRatio);

    coreParams.set_dBaseSize(dBasicBlock);
    coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
    coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
    // 向下取整保证数据量不超32K
    int64_t s1Vec2BaseSize = 8 * 1024 * 2 / (alignedD * inputDtypeBytes);
    coreParams.set_s1Vec2BaseSize(std::min(s1Vec2BaseSize, S1_VEC2_BASE_SIZE_MAX));
    coreParams.set_s1Vec2BaseTailSize(s1Size % coreParams.get_s1Vec2BaseSize());
    SetMultiBatchCoreParams();
}

void FusedFloydAttentionTilingBase::SetMultiBatchCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_bBaseSize(1);
    coreParams.set_bBaseTailSize(1);
    coreParams.set_bOuterSize(bSize);

    coreParams.set_n2BaseSize(1);
    coreParams.set_n2BaseTailSize(1);
    coreParams.set_n2OuterSize(n2Size);

    coreParams.set_gBaseSize(1);
    coreParams.set_gBaseTailSize(1);
    coreParams.set_gOuterSize(gSize);
}

void FusedFloydAttentionTilingBase::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData.multiCoreParams;
    auto &coreParams = tilingData.coreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

ge::graphStatus FusedFloydAttentionTilingBase::DoLibApiTiling()
{
    if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, batchBasic)) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    SetDataCopyTransposeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = optiling::CalcTschBlockDim(tilingData.multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    auto &inputParams = tilingData.inputParams;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (inputParams.get_needDropMaskOp() == 1) {
        blockDim = optiling::CalcTschBlockDim(aivNum, aicNum, aivNum);
        context_->SetBlockDim(blockDim);

        int64_t shapeTotalSize = inputParams.get_bSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                                 inputParams.get_s1Size() * inputParams.get_s2Size();
        auto layoutType = tilingData.inputParams.get_layoutType();
        if (layoutType == LAYOUT_TND) {
            for (int64_t i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParams.get_n2Size() * inputParams.get_gSize() * dropTotalSize;
        }
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);
        workspaces[0] += static_cast<size_t>(shapeTotalSize);
    }

    if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        tilingData.coreParams.set_pseAlibiBaseS1(pseAlibiBaseS1);
        tilingData.coreParams.set_pseAlibiBaseS2(pseAlibiBaseS2);
        int64_t pseAlibiBytes = AlignUp(pseAlibiBaseS2 * pseAlibiBaseS1 * 2, GM_ALIGN) *
                                tilingData.multiCoreParams.get_coreNum();
        workspaces[0] += pseAlibiBytes;
    }
    OPS_LOG_D(context_, "[%s] final workspace size:%zu, pseAlibiBaseS1:%ld, pseAlibiBaseS2:%ld.",
              templateName, workspaces[0], pseAlibiBaseS1, pseAlibiBaseS2);
    OPS_LOG_D_FULL(opName, "[%s] tiling data:%s", templateName, GetTilingDataDebugStr().c_str());
    OPS_LOG_D(context_, "[%s] tiling data size: %zu", templateName, tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool FusedFloydAttentionTilingBase::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                                                       matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
    bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
    bmm1.SetBias(false);
    if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
        return false;
    }

    return true;
}

bool FusedFloydAttentionTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch,
                                                    matmul_tiling::MatmulApiTiling &bmm1,
                                                    matmul_tiling::MatmulApiTiling &bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, batch, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        OPS_LOG_E(context_, "BMM1 tiling failed.");
        return false;
    }
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        OPS_LOG_E(context_, "BMM2 tiling failed.");
        return false;
    }

    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    return true;
}

bool FusedFloydAttentionTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    } else {
        OPS_LOG_D(context_, "platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm2;
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    }
}

void FusedFloydAttentionTilingBase::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

void FusedFloydAttentionTilingBase::SetDataCopyTransposeTiling()
{
    auto &coreParams = tilingData.coreParams;
    auto transposeSrcShape = ge::Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                        coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
    auto transposeDstShape = ge::Shape({bSize, n1Size, s1Size, n1Size * dSize});
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, inputDtypeBytes, tilingData.transposeTilingData);
}

void FusedFloydAttentionTilingBase::SetTensorSizeParams()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    if (tilingData.inputParams.get_pseShapeType() == PSE_B_N2_G_S1_S2) {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    } else {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s2BasicBlock); // PSE_B_N2_G_1_S2
    }

    tensorSizeParams.set_dropMaskUbSize(dropMaskExistFlag * batchInnerSize * s1BasicBlock *
                                        AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM / inputDtypeBytes);

    if (tensorSizeParams.get_pseUbSize() > 0) {
        hasPse = true;
    }
    if (tensorSizeParams.get_dropMaskUbSize() > 0) {
        hasDropOut = true;
    }
    if (tensorSizeParams.get_attenMaskUbSize() > 0) {
        hasAttenMask = true;
    }
    if (inputDtype == ge::DT_BF16 || isHighPercision) {
        if (expectTemplate.splitS2 == 1) {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
        } else {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
        }
    }
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}


ge::graphStatus FusedFloydAttentionTilingBase::GetWorkspaceSize()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
    int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
    workspaces[0] = static_cast<size_t>((bmm1Byetes + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

class FusedFloydAttentionTilingS1Bn2gs1 : public FusedFloydAttentionTilingBase {
public:
    explicit FusedFloydAttentionTilingS1Bn2gs1(gert::TilingContext *context) : FusedFloydAttentionTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitD = 1;
        templateName = "FusedFloydAttentionS1Bn2gs1";
    }
    ~FusedFloydAttentionTilingS1Bn2gs1() override = default;

protected:
    int64_t s1Ratio = 1;
    int64_t workspaceLimit = 131072; // 8*128*128
    int64_t softmaxExtraSize = 512;
    int64_t s1dHighPerfBufferNum = 4;
    int64_t s2SizeLimitMax = 1024;
    int64_t s2SizeLimitMin = 128;
    int64_t isBasicBlockNum = 64;
    int64_t minSizeLimit = 65536; // 64 * 1024
    int64_t nRatioMax = 4;
    int64_t highPerfBlock = 128;
    int64_t l1SizeRemain = 0;
    int64_t elementSize = 4;
    int64_t nzndDataLimit = 20480; // 20 * 1024
    int64_t s2SizeNzndMinLimit = 704;
    int64_t dSizeLimit = 256;
    int64_t aicRatio = 1;
    int64_t aicRatioL1reuse = 2;
    bool enableL1Reuse = false;

    bool AnalyzeDtype() override
    {
        OPS_ERR_IF(!FusedFloydAttentionTilingBase::AnalyzeDtype(),
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze base dtype."), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        // 使能增量L1reuse条件：s2>=512且BNG>64且D<=128
        // 增量L1reuse与原始L1reuse互斥
        if ((s2Size >= L1REUSE_S2_LIMIT_512 && bSize * n2Size * gSize > L1REUSE_BNG_LIMIT_64 &&
             dSize <= L1REUSE_D_Limit) ||
            (bSize * n2Size * gSize >= L1REUSE_BNG_LIMIT_4800 && s2Size == BMM2_BASICBLOCK_K_256 &&
             dSize == L1REUSE_D_LIMIT_144)) {
            enableL1Reuse = true;
            aicRatio = aicRatioL1reuse;
        }
        // 原始L1reuse说明
        // 因为一个Cube对应两个Vector, 一共需要两份L1空间存放Bmm2的右矩阵S2 * D
        // Nz的shape需要将s2Size对齐到16来计算剩余空间
        // 如果s2SizeALign16 * dSizeAlign16大于64K，则不使能该优化
        // L1reuse入口条件逻辑为：
        // 512<=S2<=1024且S1<=3840且D=64，并在非稀疏时开启
        if ((alignedS2 >= S2_REUSE_SIZE_512 && alignedS2 <= S2_REUSE_SIZE_1024) && s1Size <= S1_REUSE_SIZE_3840 &&
            alignedD == D_SPECIFIC_SIZE && (tilingData.inputParams.get_sparseType() == 0)) {
            tilingKeyBmm2Source = CubeInputSourceEnum::L1;
            enableL1Reuse = false;
            aicRatio = 1;
            if (alignedS2 == S2_REUSE_SIZE_512) {
                nRatioMax = 1;
            }
        }
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData.multiCoreParams;
        auto &coreParams = tilingData.coreParams;
        int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                            coreParams.get_s1OuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum) * aicRatio);
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    }

    void SetCoreParams() override
    {
        SetEnableL1Reuse();
        // 稀疏场景不开启S1轴N:1配比
        FusedFloydAttentionTilingBase::SetCoreParams();
        SetMultiCoreParams();
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;
        // 对于S2 < 128且S1 16对齐的场景，bmm1输出改成NZ格式，配比设为4，提升fix pipe效率
        if (alignedS2 < s2SizeLimitMin) {
            nRatioMax = 1;
        }
        // NZND入口条件逻辑为
        // 1、S2=64时开启；
        // 2、S2非64对齐时开启；
        // 3、S2大于s2SizeNzndMinLimit，且64对齐但非128对齐，BNGS1数据量大于nzndDataLimit时开启；
        // 4、满足以上条件且D>256时，开启NZND但不改变N配比；
        if ((s2Size % S2_NZTOND_SIZE_64 != 0 || s2Size == S2_NZTOND_SIZE_64) ||
            (s2Size >= s2SizeNzndMinLimit && s2Size % S2_NZTOND_SIZE_64 == 0 && s2Size % S2_NZTOND_SIZE_128 != 0 &&
             bSize * n2Size * gSize * s1Size > nzndDataLimit)) {
            if (dSize <= dSizeLimit) {
                nRatioMax = 4;
            }
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
        // 当前能分满核，考虑增大N
        while (s1Ratio < nRatioMax && multiCoreParams.get_totalSize() > ((GetNRatio() - 1) * aivNum / GetNRatio()) &&
               s1BasicBlock * GetNRatio() < alignedS1 && GetNRatio() * s1BasicBlock * alignedS2 <= workspaceLimit) {
            s1Ratio++;
            FusedFloydAttentionTilingBase::SetCoreParams();
            // S1轴N:1使能
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseSize(s1BasicBlock * GetNRatio());
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
            SetMultiCoreParams();
        }
        // 分不满核，减小N
        while (multiCoreParams.get_totalSize() <= ((GetNRatio() - 1) * aivNum / GetNRatio()) ||
               s1BasicBlock * GetNRatio() > alignedS1 || GetNRatio() * s1BasicBlock * alignedS2 > workspaceLimit) {
            s1Ratio--;
            FusedFloydAttentionTilingBase::SetCoreParams();
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseSize(s1BasicBlock * GetNRatio());
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
        }
    }

    int64_t GetNRatio() override
    {
        return s1Ratio;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float) + softmaxExtraSize;
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = s1dHighPerfBufferNum;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(128L, alignedS1); // PERFORMANCE OPT
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(static_cast<int64_t>(tilingData.coreParams.get_s1BaseSize()), s1Size), s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
            return false;
        }
        if (tilingKeyBmm2Source == CubeInputSourceEnum::L1) {
            l1SizeRemain = aicoreParams_.l1Size - alignedS2 * alignedD * elementSize;
        } else {
            l1SizeRemain = aicoreParams_.l1Size;
        }
        if (bmm1.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(static_cast<int64_t>(tilingData.coreParams.get_s1BaseSize()), s1Size), dSize, s2Size);
        bmm2.SetOrgShape(s1Size, s2StrideSize, s2Size, s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        // 在S1=S2，S2大于S2_SPECIFIC_SIZE_928且D=64时，使用性能更亲和的BMM2基本块
        if (s1Size == s2Size && s2Size >= S2_SPECIFIC_SIZE_928 && dSize == D_SPECIFIC_SIZE &&
            tilingData.inputParams.get_sparseType() == 0) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_N_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        } else {
            if (bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock) != 0) {
                return false;
            }
        }
        return true;
    }

    bool IsTemplateMatched() const override
    {
        if (s2Size > s2SizeLimitMax) {
            return false;
        }
        if (s2Size > s2SizeLimitMin) {
            return true;
        }
        if (static_cast<uint64_t>(n2Size * gSize * ((alignedS1 + alignedS2) * dSize + alignedS2)
            * inputDtypeBytes) >= aicoreParams_.l1Size ||
            static_cast<uint64_t>(n2Size * gSize * (alignedS1 + dSize) * alignedS2
            * inputDtypeBytes) >= aicoreParams_.l1Size) {
            return true;
        }
        if (n2Size * gSize * alignedS1 * alignedS2 * inputDtypeBytes <= minSizeLimit * DATA_TYPE_FP16) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::D, AxisEnum::NONE, implMode, tilingKeyDType, tilingKeyLayout,
                             tilingKeyBmm1Format, tilingKeyBmm2Source, SparseEnum::ANY,
                             PerformanceOrientedEnum::BIG_DOUBLE_BUFFER, hasDropOut, hasAttenMask, hasPse,
                             enableL1Reuse);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        if (tilingData.inputParams.get_sparseType() == 0) {
            int32_t actualUsedAivNum = CeilDivision(tilingData.multiCoreParams.get_totalSize(),
                                                    (tilingData.multiCoreParams.get_splitFactorSize() / aicRatio));
            int32_t actualUsedAivNumMod2 = actualUsedAivNum % 2;
            if (enableL1Reuse && actualUsedAivNumMod2) {
                actualUsedAivNum++;
            }
            tilingData.multiCoreParams.set_coreNum(std::min(int32_t(aivNum), actualUsedAivNum));
        }
        tilingData.bmm1TilingData.set_shareL1Size(l1SizeRemain);
        tilingData.bmm2TilingData.set_shareL1Size(l1SizeRemain);
        auto &coreParams = tilingData.coreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Size = 0;
        int64_t bmm1AlignBytes = 0;
        int64_t s2SizeAlign16 = CeilDivision(s2Size, 16L) * 16L;
        bmm1Size = CeilDivision(coreParams.get_s1BaseSize() * s2SizeAlign16, 256L) * 256L;
        bmm1AlignBytes = bmm1Size * calcTypeSize * 2;
        // bmm1和stage1的workspace不能复用
        int64_t stage1AlignBytes = bmm1AlignBytes * inputDtypeBytes / DATA_TYPE_FP32;

        /* 计算bmm2需要用的workspace, 大小为CoreNum * s1BaseSize * alignedD (16对齐）,
         * bmm2计算完成后将数据输出在这块workspace上。
         * 这块workspace主要的作用是存放bmm2的后继输出，用来做div softmax sum和cast。 */
        int64_t bmm2AlignBytes = CeilDivision(coreParams.get_s1BaseSize() * alignedD, 256L) * 256L * calcTypeSize * 2;
        workspaces[0] = static_cast<size_t>((bmm1AlignBytes + stage1AlignBytes + bmm2AlignBytes) *
                                            tilingData.multiCoreParams.get_coreNum()) +
                        WORK_SPACE_RESERVE_SIZE;
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = alignedS2;
            pseAlibiBaseS1 = std::min(static_cast<int64_t>(coreParams.get_s1BaseSize()),
                                      UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / coreParams.get_s1BaseSize());
        }
        return ge::GRAPH_SUCCESS;
    }
};

class FusedFloydAttentionTilingB : public FusedFloydAttentionTilingBase {
public:
    explicit FusedFloydAttentionTilingB(gert::TilingContext *context) : FusedFloydAttentionTilingBase(context)
    {
        templateName = "FusedFloydAttentionB";
    }
    ~FusedFloydAttentionTilingB() override = default;

protected:
    int64_t blockBSizeLimit_ = 64 * 1024;
    int64_t blockBL2SizeLimit_ = 128 * 1024;
    int64_t blockBUBSizeLimit_ = 8 * 1024;
    int64_t maxS1BaseSize_ = 256;
    int64_t dVec2BasicBlock_ = 1;
    int64_t s1Vec2BasicBlock_ = 1;

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s2BasicBlock = alignedS2;
        s1BasicBlock = blockBUBSizeLimit_ / s2BasicBlock / FRACTAL_NUM * FRACTAL_NUM;
        s1BasicBlock = std::min(s1BasicBlock, alignedS1);
        s1BasicBlock = std::min(maxS1BaseSize_, s1BasicBlock);
        dVec2BasicBlock_ = alignedD;
        s1Vec2BasicBlock_ = blockBUBSizeLimit_ / dVec2BasicBlock_ / FRACTAL_NUM * FRACTAL_NUM *
                            DATA_TYPE_FP16 / inputDtypeBytes;
        s1Vec2BasicBlock_ = std::min(s1Vec2BasicBlock_, alignedS1);
    }

    void SetCoreParams() override
    {
        auto &coreParams = tilingData.coreParams;
        auto &inputParams = tilingData.inputParams;
        int64_t n2 = inputParams.get_n2Size();
        int64_t g = inputParams.get_gSize();
        int64_t b = inputParams.get_bSize();
        int64_t s1 = inputParams.get_s1Size();
        int64_t bIn = 1;
        coreParams.set_bBaseSize(bIn);
        coreParams.set_bBaseTailSize(CalcTailSize(b, bIn));
        coreParams.set_bOuterSize(CeilDivision(b, bIn));
        coreParams.set_s1BaseSize(s1BasicBlock);
        coreParams.set_s1BaseTailSize(CalcTailSize(s1, s1BasicBlock));
        coreParams.set_s1OuterSize(CeilDivision(s1, s1BasicBlock));
        coreParams.set_s2BaseSize(s2BasicBlock);
        coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
        coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
        coreParams.set_s1Vec2BaseSize(s1Vec2BasicBlock_);
        coreParams.set_s1Vec2BaseTailSize(CalcTailSize(s1, s1Vec2BasicBlock_));
        coreParams.set_s1Vec2OuterSize(CeilDivision(s1, s1Vec2BasicBlock_));
        coreParams.set_dBaseSize(dBasicBlock);
        coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
        coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
        batchBasic = coreParams.get_bBaseSize() * n2 * g;
        OPS_LOG_D(context_, "[b:%ld, n2:%ld, g:%ld, s1:%ld, s2:%ld, batchBasic:%ld].", b, n2, g, s1,
                  inputParams.get_s2Size(), batchBasic);
        OPS_LOG_D(context_, "[bBaseSize:%d, bBaseTailSize:%d, bOuterSize:%ld].", coreParams.get_bBaseSize(),
                  coreParams.get_bBaseTailSize(), coreParams.get_bOuterSize());
        OPS_LOG_D(context_, "[s1BaseSize:%d, s1BaseTailSize:%d, s1OuterSize:%ld].", coreParams.get_s1BaseSize(),
                  coreParams.get_s1BaseTailSize(), coreParams.get_s1OuterSize());
        OPS_LOG_D(context_, "[s1Vec2BaseSize:%d, s1Vec2BaseTailSize:%d, s1Vec2OuterSize:%ld].",
                  coreParams.get_s1Vec2BaseSize(), coreParams.get_s1Vec2BaseTailSize(),
                  coreParams.get_s1Vec2OuterSize());
        OPS_LOG_D(context_, "[s2BaseSize:%d, s2BaseTailSize:%d, s2OuterSize: %ld].", coreParams.get_s2BaseSize(),
                  coreParams.get_s2BaseTailSize(), coreParams.get_s2OuterSize());
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData.multiCoreParams;
        auto &coreParams = tilingData.coreParams;
        int64_t totalSize = coreParams.get_bOuterSize(); // 核间一共处理的Bo大小
        int64_t tempUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, tempUsedAivNum)); // 每个核处理的Bo大小
        multiCoreParams.set_splitFactorTailSize(
            CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize())); // 最后一个核处理的Bo大小
        multiCoreParams.set_coreNum(
            static_cast<int32_t>(CeilDivision(totalSize, multiCoreParams.get_splitFactorSize())));
        OPS_LOG_D(context_,
                  "[totalSize:%ld, tempUsedAivNum:%ld, splitFactorSize:%ld, splitFactorTailSize:%ld, coreNum:%d].",
                  totalSize, tempUsedAivNum, multiCoreParams.get_splitFactorSize(),
                  multiCoreParams.get_splitFactorTailSize(), multiCoreParams.get_coreNum());
    }

    void SetTensorSizeParams() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;
        tensorSizeParams.set_bmm1ResUbSize(s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_pseUbSize(pseExistFlag * s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_dropMaskUbSize(dropMaskExistFlag * s1BasicBlock *
                                            AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM /
                                            inputDtypeBytes);
        if (tensorSizeParams.get_pseUbSize() > 0) {
            hasPse = true;
        }
        if (tensorSizeParams.get_dropMaskUbSize() > 0) {
            hasDropOut = true;
        }
        if (tensorSizeParams.get_attenMaskUbSize() > 0) {
            hasAttenMask = true;
        }
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            if (expectTemplate.splitS2 == 1) {
                tensorSizeParams.set_castUbSize(s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
            } else {
                tensorSizeParams.set_castUbSize(s1BasicBlock * s2BasicBlock);
            }
        }
        tensorSizeParams.set_softmaxMaxUbSize(s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
        tensorSizeParams.set_softmaxSumUbSize(s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
        tensorSizeParams.set_softmaxExpUbSize(s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
        tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);
        tensorSizeParams.set_bmm2ResUbSize(coreParams.get_s1Vec2BaseSize() * dBasicBlock);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = alignedD;
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(s1Size, s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetALayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm1.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm1.SetCLayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm1.SetBatchNum(batch);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(s1Size, dSize, s2Size);
        bmm2.SetOrgShape(s1Size, dSize, s2Size); // consider broadcst, N same as A tensor
        bmm2.SetALayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm2.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm2.SetCLayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm2.SetBatchNum(batch);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        int64_t maxDBasicBlock = AlignDown(aicoreParams_.l0cSize / (tmpS1BasicBlock * calcTypeSize), 16UL);
        if (bmm2.SetFixSplit(tmpS1BasicBlock, std::min(maxDBasicBlock, tmpDBasicBlock)) != 0) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TILINGKEY(AxisEnum::NONE, AxisEnum::NONE, AxisEnum::B, implMode, tilingKeyDType, tilingKeyLayout,
                             SparseEnum::NONE, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER, hasDropOut, hasAttenMask,
                             hasPse);
    }

    bool IsCapable() override
    {
        auto &inputParams = tilingData.inputParams;
        int64_t n2 = inputParams.get_n2Size();
        int64_t g = inputParams.get_gSize();
        bool notMatched = false;
        if (alignedS2 > HIGH_PERF_SUPPORT_S2_BASIC) {
            notMatched = true;
        }
        if (n2 * g * alignedS1 * alignedS2 * inputDtypeBytes > blockBSizeLimit_ * DATA_TYPE_FP16) {
            notMatched = true;
        }
        if (notMatched) {
            OPS_LOG_E(context_,
                      "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], "
                      "keepProb[%f]",
                      templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(),
                      inputParams.get_n2Size(), inputParams.get_gSize(), inputParams.get_s1Size(),
                      inputParams.get_s2Size(), inputParams.get_dSize(), inputParams.get_keepProb());
            return false;
        }
        return true;
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &inputParams = tilingData.inputParams;
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_bBaseSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                             inputParams.get_s1Size() * alignedS2 * calcTypeSize * inputDtypeBytes / DATA_TYPE_FP16;
        int64_t bmm2Byetes = coreParams.get_bBaseSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                             inputParams.get_s1Size() * alignedD * calcTypeSize;
        bmm1Byetes = AlignUp(bmm1Byetes, GM_ALIGN);
        bmm2Byetes = AlignUp(bmm2Byetes, GM_ALIGN);
        size_t pingPongNum = 2;
        workspaces[0] = WORK_SPACE_RESERVE_SIZE +
                        static_cast<size_t>((bmm1Byetes + bmm2Byetes) * pingPongNum * multiCoreParams.get_coreNum());

        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = alignedS2;
            pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1BasicBlock);
        }
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock, s2BasicBlock});
        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }
};

class FusedFloydAttentionTilingS1s2Bn2gs1 : public FusedFloydAttentionTilingBase {
public:
    explicit FusedFloydAttentionTilingS1s2Bn2gs1(gert::TilingContext *context) : FusedFloydAttentionTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "FusedFloydAttentionS1s2Bn2gs1";
    }
    ~FusedFloydAttentionTilingS1s2Bn2gs1() override = default;

protected:
    int64_t s2sizeLimitMin = 1024;
    int64_t dAlignSize = 16;
    bool enableL1Reuse = false;

    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        // d轴为64
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
            s1BasicBlock = std::min(128L, alignedS1);
        }
        s2BasicBlock = std::min(128L, alignedS2);
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool IsSpecialShape()
    {
        return bSize == 8 && n1Size == 32 && n2Size == 32 && s1Size == 2048 && s2Size == 2048 && dSize == 128 &&
               preTokens == 2048 && nextTokens == 0 && inputLayout[0] == 'S' && inputLayout[1] == 'B' &&
               inputLayout[2] == 'H' && pseExistFlag == 0 && attenMaskExistFlag == 1 &&
               tilingData.inputParams.get_attenMaskShapeType() == ATTEN_1_1_1_S1_S2;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        if (dSize > L1REUSE_D_Limit) {
            OPS_LOG_D(context_, "Current condition [dSize(%ld) > L1REUSE_D_Limit(%ld)] does not enable L1Reuse", dSize,
                      L1REUSE_D_Limit);
            return;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData.inputParams.get_layoutType() == LAYOUT_BNSD &&
            !(s2Size % L1REUSE_S2_LIMIT_256 == 0 || s2Size == L1REUSE_S2_LIMIT_4032)) {
            OPS_LOG_D(context_, "Current condition [dSize(%ld) && layout(BNSD)] does not enable L1Reuse", dSize);
            return;
        }
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            enableL1Reuse = true;
            return;
        }

        if ((tilingData.inputParams.get_layoutType() == LAYOUT_BSND || tilingData.inputParams.get_layoutType() ==
            LAYOUT_BSH) && s2Size <= L1REUSE_S2_Limit_2048 && dSize <= D_SPECIFIC_SIZE &&
            bSize * n1Size <= L1REUSE_BNG_Limit) {
            OPS_LOG_D(context_, "Current condition [dSize(%ld) && layout(BSH/BSND) && BN(%ld)] does not enable L1Reuse",
                      dSize, bSize * n1Size);
            return;
        }
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (dSize > BMM1_BASICBLOCK_K_64 && dSize <= BMM1_BASICBLOCK_K_128 && inputDtypeBytes != DATA_TYPE_FP32) {
            int64_t baseM = std::min(tmpS1BasicBlock, AlignUp(s1Size, FRACTAL_NUM));
            bmm1.SetFixSplit(baseM, BMM1_BASICBLOCK_N_128, dSize);
        }

        if (IsSpecialShape()) {
            if (bmm1.SetFixSplit(BMM1_BASICBLOCK_M_128, BMM1_BASICBLOCK_N_256, BMM1_BASICBLOCK_K_64) != 0) {
                return false;
            }
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        int64_t singleM = std::min(tmpS1BasicBlock, s1Size);
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(singleM, dSize,
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size),
                         s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData.inputParams.get_layoutType() == LAYOUT_BNSD &&
            tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL) &&
            singleM >= BMM2_BASICBLOCK_M_64) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, implMode, tilingKeyDType, tilingKeyLayout,
                             tilingKeyBmm1Format, SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
                             hasDropOut, hasAttenMask, hasPse, enableL1Reuse);
    }

    bool IsCapable() override
    {
        if (s2Size > s2sizeLimitMin) {
            return true;
        }
        return false;
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    void RefreshSplitFactor()
    {
        SetEnableL1Reuse();
        if (enableL1Reuse) {
            auto &multiCoreParams = tilingData.multiCoreParams;
            int64_t totalSize = multiCoreParams.get_totalSize();
            multiCoreParams.set_splitFactorSize(
                CeilDivision(totalSize, static_cast<int64_t>(multiCoreParams.get_coreNum())) * AICAIV_RATIO_2);
            multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
        }
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        RefreshSplitFactor();

        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Bytes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;

        // dSize小于64的场景，无需切D， workspace占用较小
        if (dSize <= D_SPECIFIC_SIZE) {
            // stage1占用2倍的空间，stage2占用2倍空间
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                            SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
            // NZND场景，stage1占用3倍的空间，stage2占用2倍空间
            if (s2Size % S2_NZTOND_SIZE_64 != 0) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
            // FP32场景，stage1占用4倍的空间，stage2占用2倍空间
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                                SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
        } else {
            // 切D场景，stage1占用2倍的空间，stage2占用4倍空间
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                            SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
            // NZND场景，stage1占用3倍的空间，stage2占用4倍空间
            if (s2Size % S2_NZTOND_SIZE_64 != 0) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
            // FP32场景，stage1占用4倍的空间，stage2占用4倍空间
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                                SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
        }
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = s2sizeLimitMin;
            int64_t s2Tail = s2Size % s2sizeLimitMin;
            if (s2Tail != 0) {
                pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
            } else {
                pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            }
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1BasicBlock);
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

};


// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_TILING_TEMPLATE("FusedFloydAttention", FusedFloydAttentionTilingS1s2Bn2gs1, 96);
REGISTER_TILING_TEMPLATE("FusedFloydAttention", FusedFloydAttentionTilingS1Bn2gs1, 97);
REGISTER_TILING_TEMPLATE("FusedFloydAttention", FusedFloydAttentionTilingB, 98);
} // namespace FLOYD
} // namespace optiling