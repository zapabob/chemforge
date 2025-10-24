"""
ChemForge Main CLI

メインCLI統合
train, predict, admet, generate, optimizeコマンド統合
"""

import argparse
import sys
import os
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="ChemForge - CNS Drug Discovery Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 力価予測モデル学習
  python -m chemforge train --config config.yaml --data data.csv --output models/
  
  # 力価予測
  python -m chemforge predict --model models/model.pt --config config.yaml --smiles "CCO" --output predictions.csv
  
  # ADMET予測
  python -m chemforge admet --smiles "CCO" "CCN" --output admet_results.csv
  
  # 分子生成
  python -m chemforge generate --method vae --n-molecules 100 --output generated_molecules.csv
  
  # 分子最適化
  python -m chemforge optimize --method ga --input input.csv --output optimized_molecules.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # trainコマンド
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', required=True, help='Configuration file path')
    train_parser.add_argument('--data', required=True, help='Data file path')
    train_parser.add_argument('--output', required=True, help='Output directory')
    train_parser.add_argument('--model-type', choices=['potency', 'admet'], default='potency', help='Model type')
    
    # predictコマンド
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', help='Model file path (for potency prediction)')
    predict_parser.add_argument('--config', help='Configuration file path (for potency prediction)')
    predict_parser.add_argument('--smiles', nargs='+', help='SMILES strings')
    predict_parser.add_argument('--input', help='Input file path (CSV)')
    predict_parser.add_argument('--output', required=True, help='Output file path')
    predict_parser.add_argument('--prediction-type', choices=['potency', 'admet', 'external'], required=True, help='Prediction type')
    
    # admetコマンド
    admet_parser = subparsers.add_parser('admet', help='ADMET prediction')
    admet_parser.add_argument('--smiles', nargs='+', help='SMILES strings')
    admet_parser.add_argument('--input', help='Input file path (CSV)')
    admet_parser.add_argument('--output', required=True, help='Output file path')
    admet_parser.add_argument('--prediction-type', choices=['physicochemical', 'pharmacokinetics', 'toxicity', 'drug-likeness', 'comprehensive'], default='comprehensive', help='Prediction type')
    admet_parser.add_argument('--cache-dir', default='cache', help='Cache directory')
    
    # generateコマンド
    generate_parser = subparsers.add_parser('generate', help='Generate molecules')
    generate_parser.add_argument('--method', choices=['vae', 'rl', 'ga', 'optimize'], required=True, help='Generation method')
    generate_parser.add_argument('--n-molecules', type=int, default=100, help='Number of molecules to generate')
    generate_parser.add_argument('--input', help='Input file path (for optimization)')
    generate_parser.add_argument('--output', required=True, help='Output file path')
    generate_parser.add_argument('--target-property', default='logp', help='Target property')
    generate_parser.add_argument('--target-value', type=float, default=2.0, help='Target value')
    generate_parser.add_argument('--population-size', type=int, default=50, help='Population size (for GA)')
    generate_parser.add_argument('--n-generations', type=int, default=20, help='Number of generations (for GA)')
    generate_parser.add_argument('--n-iterations', type=int, default=100, help='Number of iterations (for optimization)')
    generate_parser.add_argument('--seed', type=int, help='Random seed')
    generate_parser.add_argument('--cache-dir', default='cache', help='Cache directory')
    
    # optimizeコマンド
    optimize_parser = subparsers.add_parser('optimize', help='Optimize molecules')
    optimize_parser.add_argument('--method', choices=['ga', 'rl', 'bayesian'], required=True, help='Optimization method')
    optimize_parser.add_argument('--input', required=True, help='Input file path (CSV)')
    optimize_parser.add_argument('--output', required=True, help='Output file path')
    optimize_parser.add_argument('--target-property', default='logp', help='Target property')
    optimize_parser.add_argument('--target-value', type=float, default=2.0, help='Target value')
    optimize_parser.add_argument('--population-size', type=int, default=50, help='Population size (for GA)')
    optimize_parser.add_argument('--n-generations', type=int, default=100, help='Number of generations (for GA)')
    optimize_parser.add_argument('--n-episodes', type=int, default=1000, help='Number of episodes (for RL)')
    optimize_parser.add_argument('--n-iterations', type=int, default=100, help='Number of iterations (for Bayesian)')
    optimize_parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate (for GA)')
    optimize_parser.add_argument('--crossover-rate', type=float, default=0.8, help='Crossover rate (for GA)')
    optimize_parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate (for RL)')
    optimize_parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate (for RL)')
    optimize_parser.add_argument('--acquisition-function', choices=['ei', 'ucb'], default='ei', help='Acquisition function (for Bayesian)')
    optimize_parser.add_argument('--seed', type=int, help='Random seed')
    optimize_parser.add_argument('--cache-dir', default='cache', help='Cache directory')
    
    # バージョン情報
    parser.add_argument('--version', action='version', version='ChemForge 1.0.0')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # コマンド実行
        if args.command == 'train':
            from chemforge.cli.train import main as train_main
            train_main()
        
        elif args.command == 'predict':
            from chemforge.cli.predict import main as predict_main
            predict_main()
        
        elif args.command == 'admet':
            from chemforge.cli.admet import main as admet_main
            admet_main()
        
        elif args.command == 'generate':
            from chemforge.cli.generate import main as generate_main
            generate_main()
        
        elif args.command == 'optimize':
            from chemforge.cli.optimize import main as optimize_main
            optimize_main()
        
        else:
            print(f"[ERROR] 未知のコマンド: {args.command}")
            return
    
    except KeyboardInterrupt:
        print("\n[INFO] ユーザーによる中断")
        sys.exit(1)
    
    except Exception as e:
        print(f"[ERROR] コマンド実行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
