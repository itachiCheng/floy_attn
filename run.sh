export ASCEND_RT_VISIBLE_DEVICES=0
SCRIPT_PATH=$(realpath "$0")
echo "Script path: $SCRIPT_PATH"
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
echo "Script directory: $SCRIPT_DIR"
export ASCEND_CUSTOM_OPP_PATH=${SCRIPT_DIR}/build/_CPack_Packages/Linux/External/CANN-custom_ops-8.2.0.0-linux.aarch64.run/packages/vendors/foly_attn

rm -rf build/
#rm -rf examples src
#scp -r  root@121.40.67.29:/home/cq/foly_attn_proj/examples ./
#scp -r  root@121.40.67.29:/home/cq/foly_attn_proj/src ./

export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
mkdir build && cd build
cmake ..
make package -j 144

./CANN-custom_ops-8.2.0.0-linux.aarch64.run --quiet
cmake .. -DTESTS_EXAMPLE_OPS_TEST=test_fused_floyd_attention
make



