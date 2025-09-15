#include <string.h>
#include "graph/types.h"
#include "aclnnInner_flash_attention_score.h"

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B,
    SOC_VERSION_ASCEND910_93,
    SOC_VERSION_ASCEND910_95,
    SOC_VERSION_ASCEND310P,
    SOC_VERSION_ASCEND310B,
    SOC_VERSION_BS9SX1A,
    SOC_VERSION_ASCEND610Lite,
    SOC_VERSION_ASCEND910_55,
    SOC_VERSION_MC61AM21A
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910_93};
uint32_t socSupportListLen = 1;

TensorDesc inputDesc0_0[12] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[12] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[12] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_3[12] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_4[12] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_5[12] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_6[12] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_7[12] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_8[12] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_9[12] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_3[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_4[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_5[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_6[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_7[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_8[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_9[4] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 12, outputDesc0_0, 4};
SupportInfo list0_1 = {inputDesc0_1, 12, outputDesc0_1, 4};
SupportInfo list0_2 = {inputDesc0_2, 12, outputDesc0_2, 4};
SupportInfo list0_3 = {inputDesc0_3, 12, outputDesc0_3, 4};
SupportInfo list0_4 = {inputDesc0_4, 12, outputDesc0_4, 4};
SupportInfo list0_5 = {inputDesc0_5, 12, outputDesc0_5, 4};
SupportInfo list0_6 = {inputDesc0_6, 12, outputDesc0_6, 4};
SupportInfo list0_7 = {inputDesc0_7, 12, outputDesc0_7, 4};
SupportInfo list0_8 = {inputDesc0_8, 12, outputDesc0_8, 4};
SupportInfo list0_9 = {inputDesc0_9, 12, outputDesc0_9, 4};
SupportInfo supportInfo0[10] = {list0_0, list0_1, list0_2, list0_3, list0_4, list0_5, list0_6, list0_7, list0_8, list0_9};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 10};

OpSocSupportInfo opSocSupportList[1] = {socSupportInfo0};
OpSupportList supportList = {opSocSupportList, 1};

[[maybe_unused]] uint32_t NNOPBASE_FlashAttentionScore = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIgnoreContinuesInput(void *executor,
                                                   const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIntArrayInput(void *executor, const aclIntArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddBoolArrayInput(void *executor, const aclBoolArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddFloatArrayInput(void *executor, const aclFloatArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicInput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicOutput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index, const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray* array, const size_t index);
extern aclnnStatus NnopbaseAddFloatArrayAttr(void *executor, const aclFloatArray* array, const size_t index);
extern aclnnStatus NnopbaseAddBoolArrayAttr(void *executor, const aclBoolArray* array, const size_t index);
extern aclnnStatus NnopbaseAddArrayAttrWithDtype(void *executor, void *array, const size_t len, const size_t elementSize, const size_t index, const NnopbaseAttrDtype dtype);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace, uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);
extern aclnnStatus NnopbaseAddScalarInput(void *executor, const aclScalar *scalar, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern aclnnStatus NnopbaseAddScalarListInput(void *executor, const aclScalarList *scalarList, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern void NnopbaseAddOpTypeId(void *executor, const uint32_t opTypeId);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name, const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern aclnnStatus NnopbaseSetRef(void *executor, const size_t inputIrIdx, const size_t outputIrIdx);
extern aclnnStatus NnopbaseGetUnContiguousTensors(void *executor, const aclTensorList **inTensors);
extern aclnnStatus NnopbaseSetUnContExecutor(void *executor, aclOpExecutor *inExe, const size_t inWsSize);
extern aclnnStatus NnopbaseGetUnContExecutor(void *executor, aclOpExecutor **inExe, size_t *inWsSize);
extern aclnnStatus NnopbaseGetRefUnContiguousTensors(void *executor, const aclTensorList **unContTensors, const aclTensorList **contTensors);
extern aclnnStatus NnopbaseSetViewCopyExecutor(void *executor, aclOpExecutor *exe);
extern aclnnStatus NnopbaseGetViewCopyExecutor(void *executor, aclOpExecutor **exe);
extern aclnnStatus NnopbaseReleaseRefContiguousTensors(void *executor, const aclTensorList **tensors);
extern void *NnopbaseGetApiFunc(const char *funcName);
using AclnnContiguousGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnViewCopyGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

#define ACLNN_SUCCESS  0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

aclnnStatus aclnnInnerFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQlenOptional,
    const aclIntArray *actualSeqKvlenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTockens,
    int64_t nextTockens,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,
    int64_t pseType,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnInnerFlashAttentionScoreTiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "FlashAttentionScore";
    char inputDesc[] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char outputDesc[] = {1, 1, 1, 1};
    char attrDesc[] = {0, 0, 0, 0, 1, 1, 0, 0, 0};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(query);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(key);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(value);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(softmaxMaxOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(softmaxSumOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(softmaxOutOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attentionOutOut);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, query, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, key, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, value, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, realShiftOptional, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dropMaskOptional, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, paddingMaskOptional, 5));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, attenMaskOptional, 6));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, prefixOptional, 7));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, actualSeqQlenOptional, 8));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, actualSeqKvlenOptional, 9));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, qStartIdxOptional, 10));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, kvStartIdxOptional, 11));
    float tmp0 = static_cast<float>(scaleValue);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp0), sizeof(float), 0, kNnopbaseFloat));
    float tmp1 = static_cast<float>(keepProb);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1), sizeof(float), 1, kNnopbaseFloat));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&preTockens), sizeof(int64_t), 2, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&nextTockens), sizeof(int64_t), 3, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&headNum), sizeof(int64_t), 4, kNnopbaseInt));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(inputLayout);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(inputLayout), strlen(inputLayout) + 1, 5, kNnopbaseString));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&innerPrecise), sizeof(int64_t), 6, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&sparseMode), sizeof(int64_t), 7, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&pseType), sizeof(int64_t), 8, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, softmaxMaxOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, softmaxSumOut, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, softmaxOutOut, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, attentionOutOut, 3));
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "query", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "key", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "value", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "realShiftOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "dropMaskOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "paddingMaskOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 6, "attenMaskOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 7, "prefixOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 8, "actualSeqQlenOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 9, "actualSeqKvlenOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 10, "qStartIdxOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 11, "kvStartIdxOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "softmaxMaxOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "softmaxSumOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "softmaxOutOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "attentionOutOut", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));

    const aclTensorList *inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(*executor, &inUnContTensors);
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    if (inUnContTensors != nullptr) {
        static AclnnContiguousGetWorkspaceSizeFunc aclnnContiguousGetWorkspaceSize = (AclnnContiguousGetWorkspaceSizeFunc)NnopbaseGetApiFunc("aclnnContiguousGetWorkspaceSize");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguousGetWorkspaceSize);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor));
    }
    NnopbaseSetUnContExecutor(*executor, aclInExecutor, inContWorkspaceSize);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    *workspaceSize += inContWorkspaceSize;
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInnerFlashAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    NnopbaseGetUnContExecutor(executor, &aclInExecutor, &inContWorkspaceSize);
    if (workspaceSize < inContWorkspaceSize) {
        NnopbaseOpLogE(ACLNN_ERR_PARAM_INVALID, "input workspaceSize must be larger than contiguous size!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    workspaceSize -= inContWorkspaceSize;
    void *inWorkspace = (char *)workspace + workspaceSize;
    if (aclInExecutor != nullptr) {
        static AclnnFunc aclnnContiguous = (AclnnFunc)NnopbaseGetApiFunc("aclnnContiguous");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguous);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
