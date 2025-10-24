"""
ONNXエクスポートスクリプト
"""

import argparse
import torch
import torch.onnx
import yaml
import os
import sys
from pathlib import Path

# パス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PWA_PET_Transformer
from src.utils import get_device, load_checkpoint


def parse_args():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(description='Export MNIST PWA+PET Transformer to ONNX')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch size')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """設定ファイル読み込み"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_checkpoint(checkpoint_path: str, config: dict) -> PWA_PET_Transformer:
    """チェックポイントからモデル作成"""
    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # モデル設定をチェックポイントから取得
    if 'model_info' in checkpoint:
        model_info = checkpoint['model_info']
    else:
        # デフォルト設定
        model_info = {
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 8,
            'use_cls': True,
            'baseline': False,
            'pwa_only': False
        }
    
    # バケット設定
    buckets = config['model'].get('buckets', {"trivial": 2, "fund": 4, "adj": 2})
    
    # モデル作成
    model = PWA_PET_Transformer(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        d_model=model_info['d_model'],
        n_layers=model_info['n_layers'],
        n_heads=model_info['n_heads'],
        d_ff=config['model']['d_ff'],
        buckets=buckets,
        dropout=0.0,
        use_cls=model_info['use_cls'],
        use_rope=config['model']['rope'],
        use_pet=not model_info.get('baseline', False) and not model_info.get('pwa_only', False),
        pet_curv_reg=config['model'].get('pet_curv_reg', 1e-5),
        baseline=model_info.get('baseline', False),
        pwa_only=model_info.get('pwa_only', False)
    )
    
    return model


def export_to_onnx(
    model: PWA_PET_Transformer,
    output_path: str,
    batch_size: int = 1,
    dynamic_batch: bool = False
):
    """モデルをONNXにエクスポート"""
    model.eval()
    
    # サンプル入力
    sample_input = torch.randn(batch_size, 1, 28, 28)
    
    # 動的バッチサイズ設定
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    
    print(f"Exporting model to ONNX...")
    print(f"Output path: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Dynamic batch: {dynamic_batch}")
    
    try:
        # ONNXエクスポート
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print("✓ ONNX export successful!")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"ONNX file size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {str(e)}")
        raise


def verify_onnx_model(onnx_path: str, batch_size: int = 1):
    """ONNXモデルの検証"""
    try:
        import onnx
        import onnxruntime as ort
        
        print("Verifying ONNX model...")
        
        # ONNXモデル読み込み
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
        
        # ONNX Runtime で推論テスト
        session = ort.InferenceSession(onnx_path)
        
        # 入力形状確認
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        
        # 推論テスト
        sample_input = torch.randn(batch_size, 1, 28, 28).numpy()
        outputs = session.run(None, {'input': sample_input})
        
        print(f"ONNX inference successful! Output shape: {outputs[0].shape}")
        print("✓ ONNX Runtime verification passed")
        
    except ImportError:
        print("Warning: onnx or onnxruntime not installed. Skipping verification.")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {str(e)}")


def main():
    """メイン関数"""
    args = parse_args()
    
    # 設定読み込み
    config = load_config(args.config)
    
    # デバイス取得
    device = get_device()
    print(f"Using device: {device}")
    
    # モデル作成
    print("Creating model from checkpoint...")
    model = create_model_from_checkpoint(args.checkpoint, config)
    model = model.to(device)
    
    # チェックポイント読み込み
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # モデル情報
    model_info = model.get_model_info()
    model_name = "Baseline" if model_info.get('baseline', False) else \
                 "PWA Only" if model_info.get('pwa_only', False) else "PWA+PET"
    print(f"Model: {model_name}")
    print(f"Model info: {model_info}")
    
    # 出力パス設定
    if args.output:
        output_path = args.output
    else:
        # デフォルト出力パス
        checkpoint_name = Path(args.checkpoint).stem
        output_path = f"{checkpoint_name}_exported.onnx"
    
    # ONNXエクスポート
    export_to_onnx(
        model,
        output_path,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch
    )
    
    # 検証
    verify_onnx_model(output_path, args.batch_size)
    
    print(f"\nONNX export completed successfully!")
    print(f"Output file: {output_path}")
    
    # 使用方法の例
    print(f"\nUsage example:")
    print(f"```python")
    print(f"import onnxruntime as ort")
    print(f"import numpy as np")
    print(f"")
    print(f"# Load ONNX model")
    print(f"session = ort.InferenceSession('{output_path}')")
    print(f"")
    print(f"# Prepare input (batch_size=1, channels=1, height=28, width=28)")
    print(f"input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)")
    print(f"")
    print(f"# Run inference")
    print(f"outputs = session.run(None, {{'input': input_data}})")
    print(f"predictions = outputs[0]")
    print(f"```")


if __name__ == "__main__":
    main()
