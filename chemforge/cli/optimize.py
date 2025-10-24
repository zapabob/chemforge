"""
ChemForge CLI - Optimize Command

分子最適化コマンド実装
遺伝的アルゴリズム、強化学習、ベイズ最適化
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

class MolecularOptimizer:
    """分子最適化クラス"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = cache_dir
        self.featurizer = MolecularFeaturizer()
        self.api_manager = ExternalAPIManager(cache_dir)
    
    def optimize_genetic_algorithm(self, input_smiles: List[str], target_property: str = "logp",
                                  target_value: float = 2.0, population_size: int = 50,
                                  n_generations: int = 100, mutation_rate: float = 0.1,
                                  crossover_rate: float = 0.8, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        遺伝的アルゴリズム最適化
        
        Args:
            input_smiles: 入力SMILESリスト
            target_property: ターゲット物性
            target_value: ターゲット値
            population_size: 個体数
            n_generations: 世代数
            mutation_rate: 変異率
            crossover_rate: 交叉率
            seed: ランダムシード
            
        Returns:
            最適化結果リスト
        """
        print("=" * 60)
        print("ChemForge 遺伝的アルゴリズム最適化開始")
        print("=" * 60)
        print(f"[INFO] 入力分子数: {len(input_smiles)}")
        print(f"[INFO] ターゲット: {target_property} = {target_value}")
        print(f"[INFO] 個体数: {population_size}, 世代数: {n_generations}")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        results = []
        for i, smiles in enumerate(input_smiles):
            try:
                print(f"[INFO] 分子 {i+1}/{len(input_smiles)} 最適化開始: {smiles}")
                
                # 初期評価
                initial_properties = self._calculate_properties(smiles)
                initial_value = initial_properties.get(target_property, 0.0)
                initial_fitness = self._calculate_fitness(initial_value, target_value)
                
                # 初期個体群生成
                population = []
                for j in range(population_size):
                    individual = {
                        'smiles': self._mutate_smiles(smiles),
                        'fitness': 0.0,
                        'properties': {}
                    }
                    
                    if self._validate_smiles(individual['smiles']):
                        properties = self._calculate_properties(individual['smiles'])
                        fitness = self._calculate_fitness(properties.get(target_property, 0.0), target_value)
                        individual['fitness'] = fitness
                        individual['properties'] = properties
                    
                    population.append(individual)
                
                # 進化
                best_individual = max(population, key=lambda x: x['fitness'])
                
                for generation in range(n_generations):
                    # 選択
                    selected = self._tournament_selection(population, population_size // 2)
                    
                    # 交叉・変異
                    new_population = []
                    for j in range(0, len(selected), 2):
                        if j + 1 < len(selected):
                            parent1 = selected[j]
                            parent2 = selected[j + 1]
                            
                            # 交叉
                            if np.random.random() < crossover_rate:
                                child1, child2 = self._crossover_smiles(parent1, parent2)
                            else:
                                child1, child2 = parent1.copy(), parent2.copy()
                            
                            # 変異
                            if np.random.random() < mutation_rate:
                                child1['smiles'] = self._mutate_smiles(child1['smiles'])
                            if np.random.random() < mutation_rate:
                                child2['smiles'] = self._mutate_smiles(child2['smiles'])
                            
                            # 評価
                            for child in [child1, child2]:
                                if self._validate_smiles(child['smiles']):
                                    properties = self._calculate_properties(child['smiles'])
                                    fitness = self._calculate_fitness(properties.get(target_property, 0.0), target_value)
                                    child['fitness'] = fitness
                                    child['properties'] = properties
                                    
                                    if fitness > best_individual['fitness']:
                                        best_individual = child.copy()
                                
                                new_population.append(child)
                    
                    # 次世代更新
                    population = new_population[:population_size]
                    
                    # 進捗表示
                    if (generation + 1) % 20 == 0:
                        print(f"[INFO] 世代 {generation + 1}: 最良適応度 = {best_individual['fitness']:.4f}")
                
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': best_individual['smiles'],
                    'target_property': target_property,
                    'target_value': target_value,
                    'original_value': initial_value,
                    'optimized_value': best_individual['properties'].get(target_property, 0.0),
                    'fitness_improvement': best_individual['fitness'] - initial_fitness,
                    'optimization_method': 'genetic_algorithm',
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': None,
                    'optimization_method': 'genetic_algorithm',
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] 遺伝的アルゴリズム最適化完了: {len(results)} results")
        return results
    
    def optimize_reinforcement_learning(self, input_smiles: List[str], target_property: str = "logp",
                                      target_value: float = 2.0, n_episodes: int = 1000,
                                      learning_rate: float = 0.01, epsilon: float = 0.1,
                                      seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        強化学習最適化
        
        Args:
            input_smiles: 入力SMILESリスト
            target_property: ターゲット物性
            target_value: ターゲット値
            n_episodes: エピソード数
            learning_rate: 学習率
            epsilon: 探索率
            seed: ランダムシード
            
        Returns:
            最適化結果リスト
        """
        print("=" * 60)
        print("ChemForge 強化学習最適化開始")
        print("=" * 60)
        print(f"[INFO] 入力分子数: {len(input_smiles)}")
        print(f"[INFO] ターゲット: {target_property} = {target_value}")
        print(f"[INFO] エピソード数: {n_episodes}")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        results = []
        for i, smiles in enumerate(input_smiles):
            try:
                print(f"[INFO] 分子 {i+1}/{len(input_smiles)} 最適化開始: {smiles}")
                
                # 初期評価
                initial_properties = self._calculate_properties(smiles)
                initial_value = initial_properties.get(target_property, 0.0)
                initial_reward = self._calculate_reward(initial_value, target_value)
                
                # 強化学習最適化
                best_smiles = smiles
                best_reward = initial_reward
                best_properties = initial_properties
                
                for episode in range(n_episodes):
                    # 行動選択（ε-greedy）
                    if np.random.random() < epsilon:
                        # 探索
                        action_smiles = self._mutate_smiles(smiles)
                    else:
                        # 活用
                        action_smiles = self._mutate_smiles(best_smiles)
                    
                    if self._validate_smiles(action_smiles):
                        properties = self._calculate_properties(action_smiles)
                        reward = self._calculate_reward(properties.get(target_property, 0.0), target_value)
                        
                        # 最良解更新
                        if reward > best_reward:
                            best_smiles = action_smiles
                            best_reward = reward
                            best_properties = properties
                    
                    # 進捗表示
                    if (episode + 1) % 200 == 0:
                        print(f"[INFO] エピソード {episode + 1}: 最良報酬 = {best_reward:.4f}")
                
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': best_smiles,
                    'target_property': target_property,
                    'target_value': target_value,
                    'original_value': initial_value,
                    'optimized_value': best_properties.get(target_property, 0.0),
                    'reward_improvement': best_reward - initial_reward,
                    'optimization_method': 'reinforcement_learning',
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': None,
                    'optimization_method': 'reinforcement_learning',
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] 強化学習最適化完了: {len(results)} results")
        return results
    
    def optimize_bayesian(self, input_smiles: List[str], target_property: str = "logp",
                         target_value: float = 2.0, n_iterations: int = 100,
                         acquisition_function: str = "ei", seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        ベイズ最適化
        
        Args:
            input_smiles: 入力SMILESリスト
            target_property: ターゲット物性
            target_value: ターゲット値
            n_iterations: 反復回数
            acquisition_function: 獲得関数
            seed: ランダムシード
            
        Returns:
            最適化結果リスト
        """
        print("=" * 60)
        print("ChemForge ベイズ最適化開始")
        print("=" * 60)
        print(f"[INFO] 入力分子数: {len(input_smiles)}")
        print(f"[INFO] ターゲット: {target_property} = {target_value}")
        print(f"[INFO] 反復回数: {n_iterations}")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        results = []
        for i, smiles in enumerate(input_smiles):
            try:
                print(f"[INFO] 分子 {i+1}/{len(input_smiles)} 最適化開始: {smiles}")
                
                # 初期評価
                initial_properties = self._calculate_properties(smiles)
                initial_value = initial_properties.get(target_property, 0.0)
                initial_fitness = self._calculate_fitness(initial_value, target_value)
                
                # ベイズ最適化
                best_smiles = smiles
                best_fitness = initial_fitness
                best_properties = initial_properties
                
                for iteration in range(n_iterations):
                    # 獲得関数に基づく次の点選択
                    if acquisition_function == "ei":
                        # Expected Improvement
                        next_smiles = self._select_next_point_ei(best_smiles, target_value)
                    elif acquisition_function == "ucb":
                        # Upper Confidence Bound
                        next_smiles = self._select_next_point_ucb(best_smiles, target_value)
                    else:
                        # ランダム選択
                        next_smiles = self._mutate_smiles(best_smiles)
                    
                    if self._validate_smiles(next_smiles):
                        properties = self._calculate_properties(next_smiles)
                        fitness = self._calculate_fitness(properties.get(target_property, 0.0), target_value)
                        
                        if fitness > best_fitness:
                            best_smiles = next_smiles
                            best_fitness = fitness
                            best_properties = properties
                    
                    # 進捗表示
                    if (iteration + 1) % 20 == 0:
                        print(f"[INFO] 反復 {iteration + 1}: 最良適応度 = {best_fitness:.4f}")
                
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': best_smiles,
                    'target_property': target_property,
                    'target_value': target_value,
                    'original_value': initial_value,
                    'optimized_value': best_properties.get(target_property, 0.0),
                    'fitness_improvement': best_fitness - initial_fitness,
                    'optimization_method': 'bayesian',
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'original_smiles': smiles,
                    'optimized_smiles': None,
                    'optimization_method': 'bayesian',
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] ベイズ最適化完了: {len(results)} results")
        return results
    
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
    
    def _calculate_fitness(self, actual_value: float, target_value: float) -> float:
        """適応度計算"""
        return -abs(actual_value - target_value)
    
    def _calculate_reward(self, actual_value: float, target_value: float) -> float:
        """報酬計算"""
        return -abs(actual_value - target_value)
    
    def _validate_smiles(self, smiles: str) -> bool:
        """SMILES検証"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _mutate_smiles(self, smiles: str) -> str:
        """SMILES変異"""
        # 簡易変異（ランダムSMILES生成）
        atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
        bonds = ['', '=', '#']
        
        mutated_smiles = ""
        for _ in range(np.random.randint(5, 20)):
            atom = np.random.choice(atoms)
            bond = np.random.choice(bonds)
            mutated_smiles += bond + atom
        
        return mutated_smiles
    
    def _tournament_selection(self, population: List[Dict], n_selected: int) -> List[Dict]:
        """トーナメント選択"""
        selected = []
        for _ in range(n_selected):
            # ランダムに2個体選択
            tournament = np.random.choice(population, size=2, replace=False)
            # 適応度の高い方を選択
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        return selected
    
    def _crossover_smiles(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """SMILES交叉"""
        # 簡易交叉（ランダムSMILES生成）
        child1 = {'smiles': self._mutate_smiles(parent1['smiles'])}
        child2 = {'smiles': self._mutate_smiles(parent2['smiles'])}
        return child1, child2
    
    def _select_next_point_ei(self, current_smiles: str, target_value: float) -> str:
        """Expected Improvement選択"""
        # 簡易EI（ランダム選択）
        return self._mutate_smiles(current_smiles)
    
    def _select_next_point_ucb(self, current_smiles: str, target_value: float) -> str:
        """Upper Confidence Bound選択"""
        # 簡易UCB（ランダム選択）
        return self._mutate_smiles(current_smiles)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ChemForge Optimize Command")
    parser.add_argument("--method", choices=["ga", "rl", "bayesian"], required=True, help="最適化手法")
    parser.add_argument("--input", required=True, help="入力ファイルパス（CSV）")
    parser.add_argument("--output", required=True, help="出力ファイルパス")
    parser.add_argument("--target-property", default="logp", help="ターゲット物性")
    parser.add_argument("--target-value", type=float, default=2.0, help="ターゲット値")
    parser.add_argument("--population-size", type=int, default=50, help="個体数（GA用）")
    parser.add_argument("--n-generations", type=int, default=100, help="世代数（GA用）")
    parser.add_argument("--n-episodes", type=int, default=1000, help="エピソード数（RL用）")
    parser.add_argument("--n-iterations", type=int, default=100, help="反復回数（ベイズ用）")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="変異率（GA用）")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="交叉率（GA用）")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="学習率（RL用）")
    parser.add_argument("--epsilon", type=float, default=0.1, help="探索率（RL用）")
    parser.add_argument("--acquisition-function", choices=["ei", "ucb"], default="ei", help="獲得関数（ベイズ用）")
    parser.add_argument("--seed", type=int, help="ランダムシード")
    parser.add_argument("--cache-dir", default="cache", help="キャッシュディレクトリ")
    
    args = parser.parse_args()
    
    # 入力ファイル読み込み
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
    
    print(f"[INFO] 最適化対象: {len(input_smiles)} molecules")
    
    # 最適化実行
    optimizer = MolecularOptimizer(cache_dir=args.cache_dir)
    
    if args.method == "ga":
        results = optimizer.optimize_genetic_algorithm(
            input_smiles=input_smiles,
            target_property=args.target_property,
            target_value=args.target_value,
            population_size=args.population_size,
            n_generations=args.n_generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            seed=args.seed
        )
    
    elif args.method == "rl":
        results = optimizer.optimize_reinforcement_learning(
            input_smiles=input_smiles,
            target_property=args.target_property,
            target_value=args.target_value,
            n_episodes=args.n_episodes,
            learning_rate=args.learning_rate,
            epsilon=args.epsilon,
            seed=args.seed
        )
    
    elif args.method == "bayesian":
        results = optimizer.optimize_bayesian(
            input_smiles=input_smiles,
            target_property=args.target_property,
            target_value=args.target_value,
            n_iterations=args.n_iterations,
            acquisition_function=args.acquisition_function,
            seed=args.seed
        )
    
    else:
        print(f"[ERROR] 未知の最適化手法: {args.method}")
        return
    
    # 結果保存
    try:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"[INFO] 結果保存完了: {args.output}")
        
        # 統計表示
        success_count = len([r for r in results if r.get('status') == 'success'])
        print(f"[INFO] 成功数: {success_count}/{len(results)}")
        
        # 最適化統計
        if success_count > 0:
            improvements = [r.get('fitness_improvement', 0) for r in results if r.get('status') == 'success']
            if improvements:
                avg_improvement = np.mean(improvements)
                max_improvement = np.max(improvements)
                print(f"[INFO] 平均改善: {avg_improvement:.4f}")
                print(f"[INFO] 最大改善: {max_improvement:.4f}")
        
    except Exception as e:
        print(f"[ERROR] 結果保存失敗: {e}")

if __name__ == "__main__":
    main()