#!/bin/bash
# ============================================================
# TS-JEPA eval_autoregressive.py — mask_ratio 循环扫描脚本
# 运行日志同步写入 txt 文件
# ============================================================

LOG_FILE="/data_p6/zjx/TS_JEPA/logs/eval_autoregressive_mask_sweep_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p /data_p6/zjx/TS_JEPA/logs

echo "============================================================" | tee -a "$LOG_FILE"
echo "  TS-JEPA eval_autoregressive mask_ratio 扫描实验"           | tee -a "$LOG_FILE"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"                    | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

for MASK in 0.1 0.2 0.3 0.5 0.7 0.75 0.9
do
    echo ""                                                              | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "  [开始] mask_ratio = $MASK"                                  | tee -a "$LOG_FILE"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"                        | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    python /data_p6/zjx/TS_JEPA/eval_autoregressive.py \
        --data weather \
        --batch_size 32 \
        --lr 1e-04 \
        --lr_pretrain 1e-07 \
        --mask_ratio $MASK \
        --ema_pretrain 0.998 \
        --ratio_patches 10 \
        --checkpoint_to_use 5000 \
        --pretrain_encoder_embed_dim 128 \
        --pretrain_encoder_nhead 2 \
        --pretrain_encoder_num_layers 1 \
        --pretrain_encoder_kernel_size 3 \
        --pretrain_decoder_embed_dim 128 \
        --pretrain_decoder_nhead 2 \
        --pretrain_decoder_num_layers 1 \
        2>&1 | tee -a "$LOG_FILE"

    # 捕获退出码，判断是否成功
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  [✓ 完成] mask_ratio = $MASK | 退出码: $EXIT_CODE" | tee -a "$LOG_FILE"
    else
        echo "  [✗ 失败] mask_ratio = $MASK | 退出码: $EXIT_CODE" | tee -a "$LOG_FILE"
        echo "  → 跳过当前 mask_ratio，继续下一个..."              | tee -a "$LOG_FILE"
    fi

    echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"               | tee -a "$LOG_FILE"

done

echo ""                                                              | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  全部扫描完成！"                                             | tee -a "$LOG_FILE"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"                    | tee -a "$LOG_FILE"
echo "  日志已保存至: $LOG_FILE"                                    | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"