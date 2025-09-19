rm -rf build/
#rm -rf examples src
#scp -r  root@121.40.67.29:/home/cq/foly_attn_proj/examples ./
#scp -r  root@121.40.67.29:/home/cq/foly_attn_proj/src ./


mkdir build && cd build
cmake ..
make package -j 144

./CANN-custom_ops-8.2.0.0-linux.aarch64.run --quiet
cmake .. -DTESTS_EXAMPLE_OPS_TEST=test_fused_floyd_attention
make



