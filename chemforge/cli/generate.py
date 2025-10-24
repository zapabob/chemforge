"""
ChemForge CLI - Generate Command

分子生成コマンド実装
VAE、RL、GAによる分子生成・最適化
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

from chemforge.potency.featurizer import MolecularFeaturizer
from chemforge.utils.external_apis import ExternalAPIManager

class MolecularGenerator:
    """分子生成クラス"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = cache_dir
        self.featurizer = MolecularFeaturizer()
        self.api_manager = ExternalAPIManager(cache_dir)
    
    def generate_vae(self, n_molecules: int = 100, latent_dim: int = 128, 
                    seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        VAE分子生成
        
        Args:
            n_molecules: 生成分子数
            latent_dim: 潜在次元
            seed: ランダムシード
            
        Returns:
            生成分子リスト
        """
        print(f"[INFO] VAE分子生成開始: {n_molecules} molecules")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        results = []
        for i in range(n_molecules):
            try:
                # 簡易VAE生成（ランダムSMILES生成）
                smiles = self._generate_random_smiles()
                
                # 分子検証
                if self._validate_smiles(smiles):
                    result = {
                        'smiles': smiles,
                        'generation_method': 'vae',
                        'generation_id': i,
                        'status': 'success'
                    }
                else:
                    result = {
                        'smiles': smiles,
                        'generation_method': 'vae',
                        'generation_id': i,
                        'status': 'invalid'
                    }
                
            except Exception as e:
                result = {
                    'smiles': None,
                    'generation_method': 'vae',
                    'generation_id': i,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] VAE分子生成完了: {len(results)} results")
        return results
    
    def generate_rl(self, n_molecules: int = 100, target_property: str = "logp",
                   target_value: float = 2.0, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        RL分子生成
        
        Args:
            n_molecules: 生成分子数
            target_property: ターゲット物性
            target_value: ターゲット値
            seed: ランダムシード
            
        Returns:
            生成分子リスト
        """
        print(f"[INFO] RL分子生成開始: {n_molecules} molecules")
        print(f"[INFO] ターゲット: {target_property} = {target_value}")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        results = []
        for i in range(n_molecules):
            try:
                # 簡易RL生成（ランダムSMILES生成）
                smiles = self._generate_random_smiles()
                
                # 分子検証
                if self._validate_smiles(smiles):
                    # 物性計算
                    properties = self._calculate_properties(smiles)
                    target_prop_value = properties.get(target_property, 0.0)
                    
                    # 報酬計算
                    reward = self._calculate_reward(target_prop_value, target_value)
                    
                    result = {
                        'smiles': smiles,
                        'generation_method': 'rl',
                        'generation_id': i,
                        'target_property': target_property,
                        'target_value': target_value,
                        'actual_value': target_prop_value,
                        'reward': reward,
                        'status': 'success'
                    }
                else:
                    result = {
                        'smiles': smiles,
                        'generation_method': 'rl',
                        'generation_id': i,
                        'status': 'invalid'
                    }
                
            except Exception as e:
                result = {
                    'smiles': None,
                    'generation_method': 'rl',
                    'generation_id': i,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] RL分子生成完了: {len(results)} results")
        return results
    
    def generate_ga(self, n_molecules: int = 100, population_size: int = 50,
                   n_generations: int = 20, target_property: str = "logp",
                   target_value: float = 2.0, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        GA分子生成
        
        Args:
            n_molecules: 生成分子数
            population_size: 個体数
            n_generations: 世代数
            target_property: ターゲット物性
            target_value: ターゲット値
            seed: ランダムシード
            
        Returns:
            生成分子リスト
        """
        print(f"[INFO] GA分子生成開始: {n_molecules} molecules")
        print(f"[INFO] 個体数: {population_size}, 世代数: {n_generations}")
        print(f"[INFO] ターゲット: {target_property} = {target_value}")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 初期個体生成
        population = []
        for i in range(population_size):
            smiles = self._generate_random_smiles()
            if self._validate_smiles(smiles):
                properties = self._calculate_properties(smiles)
                fitness = self._calculate_fitness(properties.get(target_property, 0.0), target_value)
                population.append({
                    'smiles': smiles,
                    'fitness': fitness,
                    'properties': properties
                })
        
        # 進化
        for generation in range(n_generations):
            print(f"[INFO] 世代 {generation + 1}/{n_generations}")
            
            # 選択
            selected = self._selection(population, population_size // 2)
            
            # 交叉・変異
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1 = selected[i]
                    parent2 = selected[i + 1]
                    
                    # 交叉
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    # 変異
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    
                    # 評価
                    for child in [child1, child2]:
                        if child['smiles'] and self._validate_smiles(child['smiles']):
                            properties = self._calculate_properties(child['smiles'])
                            fitness = self._calculate_fitness(properties.get(target_property, 0.0), target_value)
                            child['fitness'] = fitness
                            child['properties'] = properties
                            new_population.append(child)
            
            # 次世代更新
            population = new_population[:population_size]
        
        # 結果整理
        results = []
        for i, individual in enumerate(population):
            result = {
                'smiles': individual['smiles'],
                'generation_method': 'ga',
                'generation_id': i,
                'target_property': target_property,
                'target_value': target_value,
                'actual_value': individual['properties'].get(target_property, 0.0),
                'fitness': individual['fitness'],
                'status': 'success'
            }
            results.append(result)
        
        print(f"[INFO] GA分子生成完了: {len(results)} results")
        return results
    
    def optimize_molecules(self, input_smiles: List[str], target_property: str = "logp",
                          target_value: float = 2.0, n_iterations: int = 100) -> List[Dict[str, Any]]:
        """
        分子最適化
        
        Args:
            input_smiles: 入力SMILESリスト
            target_property: ターゲット物性
            target_value: ターゲット値
            n_iterations: 反復回数
            
        Returns:
            最適化結果リスト
        """
        print(f"[INFO] 分子最適化開始: {len(input_smiles)} molecules")
        print(f"[INFO] ターゲット: {target_property} = {target_value}")
        
        results = []
        for i, smiles in enumerate(input_smiles):
            try:
                # 初期評価
                initial_properties = self._calculate_properties(smiles)
                initial_value = initial_properties.get(target_property, 0.0)
                initial_fitness = self._calculate_fitness(initial_value, target_value)
                
                # 最適化（簡易版）
                best_smiles = smiles
                best_fitness = initial_fitness
                best_properties = initial_properties
                
                for iteration in range(n_iterations):
                    # 変異
                    mutated_smiles = self._mutation_smiles(smiles)
                    
                    if self._validate_smiles(mutated_smiles):
                        properties = self._calculate_properties(mutated_smiles)
                        fitness = self._calculate_fitness(properties.get(target_property, 0.0), target_value)
                        
                        if fitness > best_fitness:
                            best_smiles = mutated_smiles
                            best_fitness = fitness
                            best_properties = properties
                
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': best_smiles,
                    'target_property': target_property,
                    'target_value': target_value,
                    'original_value': initial_value,
                    'optimized_value': best_properties.get(target_property, 0.0),
                    'fitness_improvement': best_fitness - initial_fitness,
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': None,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] 分子最適化完了: {len(results)} results")
        return results
    
    def _generate_random_smiles(self) -> str:
        """ランダムSMILES生成"""
        # 簡易ランダムSMILES生成
        atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
        bonds = ['', '=', '#']
        
        smiles = ""
        for _ in range(np.random.randint(5, 20)):
            atom = np.random.choice(atoms)
            bond = np.random.choice(bonds)
            smiles += bond + atom
        
        return smiles
    
    def _validate_smiles(self, smiles: str) -> bool:
        """SMILES検証"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _calculate_properties(self, smiles: str) -> Dict[str, float]:
        """物性計算"""
        try:
            features = self.featurizer.transform([{'smiles': smiles}])
            descriptors = features['descriptors'][0].cpu().numpy()
            
            return {
                'molecular_weight': float(descriptors[0]) if len(descriptors) > 0 else 0.0,
                'logp': float(descriptors[1]) if len(descriptors) > 1 else 0.0,
                'tpsa': float(descriptors[2]) if len(descriptors) > 2 else 0.0,
                'hbd': float(descriptors[3]) if len(descriptors) > 3 else 0.0,
                'hba': float(descriptors[4]) if len(descriptors) > 4 else 0.0
            }
        except:
            return {}
    
    def _calculate_reward(self, actual_value: float, target_value: float) -> float:
        """報酬計算"""
        return -abs(actual_value - target_value)
    
    def _calculate_fitness(self, actual_value: float, target_value: float) -> float:
        """適応度計算"""
        return -abs(actual_value - target_value)
    
    def _selection(self, population: List[Dict], n_selected: int) -> List[Dict]:
        """選択"""
        # 適応度順ソート
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        return sorted_pop[:n_selected]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉"""
        # 簡易交叉（ランダムSMILES生成）
        child1 = {'smiles': self._generate_random_smiles()}
        child2 = {'smiles': self._generate_random_smiles()}
        return child1, child2
    
    def _mutation(self, individual: Dict) -> Dict:
        """変異"""
        # 簡易変異（ランダムSMILES生成）
        individual['smiles'] = self._generate_random_smiles()
        return individual
    
    def _mutation_smiles(self, smiles: str) -> str:
        """SMILES変異"""
        # 簡易変異（ランダムSMILES生成）
        return self._generate_random_smiles()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ChemForge Generate Command")
    parser.add_argument("--method", choices=["vae", "rl", "ga", "optimize"], required=True, help="生成手法")
    parser.add_argument("--n-molecules", type=int, default=100, help="生成分子数")
    parser.add_argument("--input", help="入力ファイルパス（最適化用）")
    parser.add_argument("--output", required=True, help="出力ファイルパス")
    parser.add_argument("--target-property", default="logp", help="ターゲット物性")
    parser.add_argument("--target-value", type=float, default=2.0, help="ターゲット値")
    parser.add_argument("--population-size", type=int, default=50, help="個体数（GA用）")
    parser.add_argument("--n-generations", type=int, default=20, help="世代数（GA用）")
    parser.add_argument("--n-iterations", type=int, default=100, help="反復回数（最適化用）")
    parser.add_argument("--seed", type=int, help="ランダムシード")
    parser.add_argument("--cache-dir", default="cache", help="キャッシュディレクトリ")
    
    args = parser.parse_args()
    
    # 分子生成実行
    generator = MolecularGenerator(cache_dir=args.cache_dir)
    
    if args.method == "vae":
        results = generator.generate_vae(
            n_molecules=args.n_molecules,
            seed=args.seed
        )
    
    elif args.method == "rl":
        results = generator.generate_rl(
            n_molecules=args.n_molecules,
            target_property=args.target_property,
            target_value=args.target_value,
            seed=args.seed
        )
    
    elif args.method == "ga":
        results = generator.generate_ga(
            n_molecules=args.n_molecules,
            population_size=args.population_size,
            n_generations=args.n_generations,
            target_property=args.target_property,
            target_value=args.target_value,
            seed=args.seed
        )
    
    elif args.method == "optimize":
        if not args.input:
            print("[ERROR] 最適化には --input が必要です")
            return
        
        try:
            df = pd.read_csv(args.input)
            if 'smiles' in df.columns:
                input_smiles = df['smiles'].tolist()
            else:
                print("[ERROR] 入力ファイルに'smiles'列が見つかりません")
                return
        except Exception as e:
            print(f"[ERROR] 入力ファイル読み込み失敗: {e}")
            return
        
        results = generator.optimize_molecules(
            input_smiles=input_smiles,
            target_property=args.target_property,
            target_value=args.target_value,
            n_iterations=args.n_iterations
        )
    
    else:
        print(f"[ERROR] 未知の生成手法: {args.method}")
        return
    
    # 結果保存
    try:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"[INFO] 結果保存完了: {args.output}")
        
        # 統計表示
        success_count = len([r for r in results if r.get('status') == 'success'])
        print(f"[INFO] 成功数: {success_count}/{len(results)}")
        
    except Exception as e:
        print(f"[ERROR] 結果保存失敗: {e}")

if __name__ == "__main__":
    main()