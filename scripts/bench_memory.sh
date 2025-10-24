#!/bin/bash
# メモリベンチマークスクリプト

echo "=== MNIST PWA+PET Transformer Memory Benchmark ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo ""

# 設定
BATCH_SIZES=(64 128 256 512)
MODELS=("baseline" "pwa" "pwa_pet")
PATCH_SIZES=(1 2 4)

# 結果保存ディレクトリ
RESULTS_DIR="./benchmark_results"
mkdir -p $RESULTS_DIR

# ベンチマーク実行関数
run_benchmark() {
    local model_type=$1
    local batch_size=$2
    local patch_size=$3
    
    echo "Testing: $model_type, batch_size=$batch_size, patch_size=$patch_size"
    
    # モデルタイプに応じたフラグ設定
    local flags=""
    case $model_type in
        "baseline")
            flags="--baseline"
            ;;
        "pwa")
            flags="--pwa"
            ;;
        "pwa_pet")
            flags="--pwa-pet"
            ;;
    esac
    
    # パッチサイズフラグ
    if [ $patch_size -ne 2 ]; then
        flags="$flags --patch $patch_size"
    fi
    
    # バッチサイズフラグ
    flags="$flags --bs $batch_size"
    
    # 短時間訓練でメモリ使用量測定
    echo "Running memory benchmark..."
    python run_train.py $flags --config config.yaml 2>&1 | tee "${RESULTS_DIR}/${model_type}_bs${batch_size}_patch${patch_size}.log"
    
    # GPUメモリ情報取得
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> "${RESULTS_DIR}/gpu_memory.log"
    
    echo "Completed: $model_type, batch_size=$batch_size, patch_size=$patch_size"
    echo ""
}

# メイン実行
echo "Starting comprehensive memory benchmark..."
echo ""

# 各モデルタイプでテスト
for model in "${MODELS[@]}"; do
    echo "=== Testing $model model ==="
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        for patch_size in "${PATCH_SIZES[@]}"; do
            run_benchmark $model $batch_size $patch_size
            
            # メモリクリア
            python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            sleep 2
        done
    done
done

# 結果サマリー生成
echo "=== Benchmark Results Summary ==="
echo "Results saved to: $RESULTS_DIR"
echo ""

# GPUメモリ使用量サマリー
if [ -f "${RESULTS_DIR}/gpu_memory.log" ]; then
    echo "GPU Memory Usage Summary:"
    cat "${RESULTS_DIR}/gpu_memory.log" | tail -20
fi

echo ""
echo "Benchmark completed at $(date)"
