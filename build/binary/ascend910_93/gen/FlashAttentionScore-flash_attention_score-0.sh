#!/bin/bash
echo "[ascend910_93] Generating FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=flash_attention_score --input_param=/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/binary/ascend910_93/gen/FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3_param.json --soc_version=Ascend910_9391                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3.json ; then
  echo "$2/FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3.json not generated!"
  exit 1
fi

if ! test -f $2/FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3.o ; then
  echo "$2/FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3.o not generated!"
  exit 1
fi
echo "[ascend910_93] Generating FlashAttentionScore_d0fddb39abdd5c6337066422cb3103d3 Done"
