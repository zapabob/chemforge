"""
Molecular Generator Module

分子生成モジュール
VAE・RL・GAを活用した効率的な分子生成システム
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 既存モジュール活用
from chemforge.models.transformer import PotencyTransformer
from chemforge.models.gnn import PotencyGNN
from chemforge.models.ensemble import PotencyEnsemble
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class MolecularGenerator:
    """
    分子生成クラス
    
    VAE・RL・GAを活用した効率的な分子生成システム
    """
    
    def __init__(self, config_path: Optional[str] = None, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            cache_dir: キャッシュディレクトリ
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 既存モジュール活用
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.logger = Logger("MolecularGenerator")
        self.validator = DataValidator()
        
        # 分子特徴量・記述子
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        self.rdkit_descriptors = RDKitDescriptors(config_path, cache_dir)
        
        # 生成設定
        self.generation_config = self.config.get('molecular_generation', {})
        self.vae_config = self.generation_config.get('vae', {})
        self.rl_config = self.generation_config.get('rl', {})
        self.ga_config = self.generation_config.get('ga', {})
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("MolecularGenerator initialized")
    
    def generate_molecules(self, num_molecules: int = 1000,
                          method: str = 'vae',
                          target_properties: Optional[Dict] = None,
                          seed_molecules: Optional[List[str]] = None) -> List[str]:
        """
        分子生成
        
        Args:
            num_molecules: 生成分子数
            method: 生成方法（vae, rl, ga）
            target_properties: ターゲット特性
            seed_molecules: シード分子
            
        Returns:
            生成分子SMILESリスト
        """
        logger.info(f"Generating {num_molecules} molecules using {method} method")
        
        if method == 'vae':
            molecules = self._generate_with_vae(num_molecules, target_properties, seed_molecules)
        elif method == 'rl':
            molecules = self._generate_with_rl(num_molecules, target_properties, seed_molecules)
        elif method == 'ga':
            molecules = self._generate_with_ga(num_molecules, target_properties, seed_molecules)
        else:
            raise ValueError(f"Unsupported generation method: {method}")
        
        logger.info(f"Generated {len(molecules)} molecules using {method}")
        return molecules
    
    def _generate_with_vae(self, num_molecules: int,
                          target_properties: Optional[Dict],
                          seed_molecules: Optional[List[str]]) -> List[str]:
        """
        VAE生成
        
        Args:
            num_molecules: 生成分子数
            target_properties: ターゲット特性
            seed_molecules: シード分子
            
        Returns:
            生成分子SMILESリスト
        """
        from chemforge.generation.vae_generator import VAEGenerator
        
        vae_generator = VAEGenerator(self.config_path, self.cache_dir)
        molecules = vae_generator.generate_molecules(
            num_molecules=num_molecules,
            target_properties=target_properties,
            seed_molecules=seed_molecules
        )
        
        return molecules
    
    def _generate_with_rl(self, num_molecules: int,
                         target_properties: Optional[Dict],
                         seed_molecules: Optional[List[str]]) -> List[str]:
        """
        RL生成
        
        Args:
            num_molecules: 生成分子数
            target_properties: ターゲット特性
            seed_molecules: シード分子
            
        Returns:
            生成分子SMILESリスト
        """
        from chemforge.generation.rl_generator import RLGenerator
        
        rl_generator = RLGenerator(self.config_path, self.cache_dir)
        molecules = rl_generator.generate_molecules(
            num_molecules=num_molecules,
            target_properties=target_properties,
            seed_molecules=seed_molecules
        )
        
        return molecules
    
    def _generate_with_ga(self, num_molecules: int,
                         target_properties: Optional[Dict],
                         seed_molecules: Optional[List[str]]) -> List[str]:
        """
        GA生成
        
        Args:
            num_molecules: 生成分子数
            target_properties: ターゲット特性
            seed_molecules: シード分子
            
        Returns:
            生成分子SMILESリスト
        """
        from chemforge.generation.ga_optimizer import GAOptimizer
        
        ga_optimizer = GAOptimizer(self.config_path, self.cache_dir)
        molecules = ga_optimizer.generate_molecules(
            num_molecules=num_molecules,
            target_properties=target_properties,
            seed_molecules=seed_molecules
        )
        
        return molecules
    
    def optimize_molecules(self, molecules: List[str],
                          target_properties: Dict[str, float],
                          optimization_method: str = 'ga',
                          max_iterations: int = 100) -> List[str]:
        """
        分子最適化
        
        Args:
            molecules: 最適化対象分子
            target_properties: ターゲット特性
            optimization_method: 最適化方法
            max_iterations: 最大反復数
            
        Returns:
            最適化分子SMILESリスト
        """
        logger.info(f"Optimizing {len(molecules)} molecules using {optimization_method}")
        
        if optimization_method == 'ga':
            optimized_molecules = self._optimize_with_ga(
                molecules, target_properties, max_iterations
            )
        elif optimization_method == 'rl':
            optimized_molecules = self._optimize_with_rl(
                molecules, target_properties, max_iterations
            )
        else:
            raise ValueError(f"Unsupported optimization method: {optimization_method}")
        
        logger.info(f"Optimized {len(optimized_molecules)} molecules")
        return optimized_molecules
    
    def _optimize_with_ga(self, molecules: List[str],
                         target_properties: Dict[str, float],
                         max_iterations: int) -> List[str]:
        """
        GA最適化
        
        Args:
            molecules: 最適化対象分子
            target_properties: ターゲット特性
            max_iterations: 最大反復数
            
        Returns:
            最適化分子SMILESリスト
        """
        from chemforge.generation.ga_optimizer import GAOptimizer
        
        ga_optimizer = GAOptimizer(self.config_path, self.cache_dir)
        optimized_molecules = ga_optimizer.optimize_molecules(
            molecules=molecules,
            target_properties=target_properties,
            max_iterations=max_iterations
        )
        
        return optimized_molecules
    
    def _optimize_with_rl(self, molecules: List[str],
                         target_properties: Dict[str, float],
                         max_iterations: int) -> List[str]:
        """
        RL最適化
        
        Args:
            molecules: 最適化対象分子
            target_properties: ターゲット特性
            max_iterations: 最大反復数
            
        Returns:
            最適化分子SMILESリスト
        """
        from chemforge.generation.rl_generator import RLGenerator
        
        rl_generator = RLGenerator(self.config_path, self.cache_dir)
        optimized_molecules = rl_generator.optimize_molecules(
            molecules=molecules,
            target_properties=target_properties,
            max_iterations=max_iterations
        )
        
        return optimized_molecules
    
    def evaluate_molecules(self, molecules: List[str],
                          target_properties: Optional[Dict] = None) -> pd.DataFrame:
        """
        分子評価
        
        Args:
            molecules: 評価対象分子
            target_properties: ターゲット特性
            
        Returns:
            評価結果DataFrame
        """
        logger.info(f"Evaluating {len(molecules)} molecules")
        
        # 分子特徴量計算
        features_data = []
        for smiles in tqdm(molecules, desc="Calculating molecular features"):
            try:
                features = self.molecular_features.calculate_features(smiles, include_3d=True)
                features['smiles'] = smiles
                features_data.append(features)
            except Exception as e:
                logger.warning(f"Error calculating features for {smiles}: {e}")
                continue
        
        # DataFrame作成
        df = pd.DataFrame(features_data)
        
        # ターゲット特性評価
        if target_properties:
            df = self._evaluate_target_properties(df, target_properties)
        
        logger.info(f"Evaluated {len(df)} molecules")
        return df
    
    def _evaluate_target_properties(self, df: pd.DataFrame,
                                   target_properties: Dict[str, float]) -> pd.DataFrame:
        """
        ターゲット特性評価
        
        Args:
            df: 分子データフレーム
            target_properties: ターゲット特性
            
        Returns:
            評価結果DataFrame
        """
        for prop_name, target_value in target_properties.items():
            if prop_name in df.columns:
                # ターゲット値との差を計算
                df[f'{prop_name}_diff'] = abs(df[prop_name] - target_value)
                df[f'{prop_name}_score'] = 1.0 / (1.0 + df[f'{prop_name}_diff'])
            else:
                logger.warning(f"Target property {prop_name} not found in molecular features")
        
        # 総合スコア計算
        score_columns = [col for col in df.columns if col.endswith('_score')]
        if score_columns:
            df['total_score'] = df[score_columns].mean(axis=1)
            df = df.sort_values('total_score', ascending=False)
        
        return df
    
    def filter_molecules(self, molecules: List[str],
                        filters: Dict[str, Any]) -> List[str]:
        """
        分子フィルタリング
        
        Args:
            molecules: フィルタリング対象分子
            filters: フィルター条件
            
        Returns:
            フィルタリング済み分子リスト
        """
        logger.info(f"Filtering {len(molecules)} molecules")
        
        # 分子評価
        df = self.evaluate_molecules(molecules)
        
        # フィルター適用
        filtered_df = df.copy()
        
        for prop_name, filter_value in filters.items():
            if prop_name in filtered_df.columns:
                if isinstance(filter_value, dict):
                    # 範囲フィルター
                    if 'min' in filter_value:
                        filtered_df = filtered_df[filtered_df[prop_name] >= filter_value['min']]
                    if 'max' in filter_value:
                        filtered_df = filtered_df[filtered_df[prop_name] <= filter_value['max']]
                else:
                    # 値フィルター
                    filtered_df = filtered_df[filtered_df[prop_name] == filter_value]
            else:
                logger.warning(f"Filter property {prop_name} not found in molecular features")
        
        filtered_molecules = filtered_df['smiles'].tolist()
        logger.info(f"Filtered to {len(filtered_molecules)} molecules")
        
        return filtered_molecules
    
    def save_generated_molecules(self, molecules: List[str],
                               output_path: str,
                               include_features: bool = True) -> bool:
        """
        生成分子保存
        
        Args:
            molecules: 保存対象分子
            output_path: 出力パス
            include_features: 特徴量含むフラグ
            
        Returns:
            成功フラグ
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if include_features:
                # 特徴量含む保存
                df = self.evaluate_molecules(molecules)
                df.to_csv(output_path, index=False)
            else:
                # SMILESのみ保存
                df = pd.DataFrame({'smiles': molecules})
                df.to_csv(output_path, index=False)
            
            logger.info(f"Generated molecules saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving generated molecules: {e}")
            return False
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """
        生成サマリー取得
        
        Returns:
            生成サマリー辞書
        """
        summary = {
            'generation_config': self.generation_config,
            'vae_config': self.vae_config,
            'rl_config': self.rl_config,
            'ga_config': self.ga_config,
            'device': str(self.device),
            'cache_dir': str(self.cache_dir)
        }
        
        return summary

def create_molecular_generator(config_path: Optional[str] = None, 
                             cache_dir: str = "cache") -> MolecularGenerator:
    """
    分子生成器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        MolecularGenerator
    """
    return MolecularGenerator(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    molecular_generator = MolecularGenerator()
    
    print(f"MolecularGenerator created: {molecular_generator}")
    print(f"Device: {molecular_generator.device}")
    print(f"Cache directory: {molecular_generator.cache_dir}")
    print(f"Generation config: {molecular_generator.generation_config}")