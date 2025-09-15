
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_FLASH_ATTENTION_SCORE_H_
#define ACLNN_INNER_FLASH_ATTENTION_SCORE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerFlashAttentionScoreGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * realShiftOptional : optional
 * dropMaskOptional : optional
 * paddingMaskOptional : optional
 * attenMaskOptional : optional
 * prefixOptional : optional
 * actualSeqQlenOptional : optional
 * actualSeqKvlenOptional : optional
 * qStartIdxOptional : optional
 * kvStartIdxOptional : optional
 * scaleValue : optional
 * keepProb : optional
 * preTockens : optional
 * nextTockens : optional
 * headNum : required
 * inputLayout : required
 * innerPrecise : optional
 * sparseMode : optional
 * pseType : optional
 * softmaxMaxOut : required
 * softmaxSumOut : required
 * softmaxOutOut : required
 * attentionOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

/* funtion: aclnnInnerFlashAttentionScore
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFlashAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
