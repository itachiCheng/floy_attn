#!/bin/bash
echo "[ascend910_93] Generating FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0 ..."
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
res=$(opc $1 --main_func=flash_attention_score --input_param=/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/binary/ascend910_93/gen/FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0_param.json --soc_version=Ascend910_9391                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0.json ; then
  echo "$2/FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0.json not generated!"
  exit 1
fi

if ! test -f $2/FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0.o ; then
  echo "$2/FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0.o not generated!"
  exit 1
fi
echo "[ascend910_93] Generating FlashAttentionScore_5881aeec01e51adb01fb1db8be1c04f0 Done"
