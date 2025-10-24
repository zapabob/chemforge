"""
Database utilities for ChemForge platform.

This module provides database management functionality including ChEMBL database
integration, local database operations, and data persistence.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle
from contextlib import contextmanager

from chemforge.data.chembl_loader import ChEMBLLoader


class DatabaseManager:
    """Base database manager class."""
    
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to database file
            logger: Logger instance
        """
        self.db_path = Path(db_path)
        self.logger = logger or logging.getLogger(__name__)
        self.connection = None
        
    def connect(self):
        """Connect to database."""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            self.logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from database."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Disconnected from database")
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        if not self.connection:
            self.connect()
        try:
            yield self.connection
        finally:
            pass  # Don't close connection here, let disconnect() handle it
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """
        Execute SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return results
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute SQL update query and return affected rows.
        
        Args:
            query: SQL update query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount


class ChEMBLDatabase(DatabaseManager):
    """ChEMBL database manager."""
    
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize ChEMBL database manager.
        
        Args:
            db_path: Path to ChEMBL database file
            logger: Logger instance
        """
        super().__init__(db_path, logger)
        self.chembl_loader = ChEMBLLoader()
        
    def create_tables(self):
        """Create ChEMBL database tables."""
        create_queries = [
            """
            CREATE TABLE IF NOT EXISTS molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chembl_id TEXT UNIQUE,
                smiles TEXT,
                mol_weight REAL,
                logp REAL,
                hbd INTEGER,
                hba INTEGER,
                tpsa REAL,
                rotatable_bonds INTEGER,
                aromatic_rings INTEGER,
                heavy_atoms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER,
                target_id INTEGER,
                activity_type TEXT,
                activity_value REAL,
                activity_unit TEXT,
                activity_relation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (molecule_id) REFERENCES molecules (id),
                FOREIGN KEY (target_id) REFERENCES targets (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chembl_id TEXT UNIQUE,
                target_name TEXT,
                target_type TEXT,
                organism TEXT,
                uniprot_id TEXT,
                gene_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER,
                target_id INTEGER,
                prediction_value REAL,
                confidence REAL,
                model_name TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (molecule_id) REFERENCES molecules (id),
                FOREIGN KEY (target_id) REFERENCES targets (id)
            )
            """
        ]
        
        for query in create_queries:
            self.execute_update(query)
        
        self.logger.info("ChEMBL database tables created successfully")
    
    def insert_molecule(self, molecule_data: Dict[str, Any]) -> int:
        """
        Insert molecule data into database.
        
        Args:
            molecule_data: Dictionary containing molecule information
            
        Returns:
            Inserted molecule ID
        """
        query = """
        INSERT INTO molecules (chembl_id, smiles, mol_weight, logp, hbd, hba, tpsa, 
                              rotatable_bonds, aromatic_rings, heavy_atoms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            molecule_data.get('chembl_id'),
            molecule_data.get('smiles'),
            molecule_data.get('mol_weight'),
            molecule_data.get('logp'),
            molecule_data.get('hbd'),
            molecule_data.get('hba'),
            molecule_data.get('tpsa'),
            molecule_data.get('rotatable_bonds'),
            molecule_data.get('aromatic_rings'),
            molecule_data.get('heavy_atoms')
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def insert_target(self, target_data: Dict[str, Any]) -> int:
        """
        Insert target data into database.
        
        Args:
            target_data: Dictionary containing target information
            
        Returns:
            Inserted target ID
        """
        query = """
        INSERT INTO targets (chembl_id, target_name, target_type, organism, uniprot_id, gene_name)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            target_data.get('chembl_id'),
            target_data.get('target_name'),
            target_data.get('target_type'),
            target_data.get('organism'),
            target_data.get('uniprot_id'),
            target_data.get('gene_name')
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def insert_activity(self, activity_data: Dict[str, Any]) -> int:
        """
        Insert activity data into database.
        
        Args:
            activity_data: Dictionary containing activity information
            
        Returns:
            Inserted activity ID
        """
        query = """
        INSERT INTO activities (molecule_id, target_id, activity_type, activity_value, 
                               activity_unit, activity_relation)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            activity_data.get('molecule_id'),
            activity_data.get('target_id'),
            activity_data.get('activity_type'),
            activity_data.get('activity_value'),
            activity_data.get('activity_unit'),
            activity_data.get('activity_relation')
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """
        Insert prediction data into database.
        
        Args:
            prediction_data: Dictionary containing prediction information
            
        Returns:
            Inserted prediction ID
        """
        query = """
        INSERT INTO predictions (molecule_id, target_id, prediction_value, confidence, 
                               model_name, model_version)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            prediction_data.get('molecule_id'),
            prediction_data.get('target_id'),
            prediction_data.get('prediction_value'),
            prediction_data.get('confidence'),
            prediction_data.get('model_name'),
            prediction_data.get('model_version')
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def get_molecules(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get molecules from database.
        
        Args:
            limit: Maximum number of molecules to return
            
        Returns:
            DataFrame containing molecule data
        """
        query = "SELECT * FROM molecules"
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.execute_query(query)
        return pd.DataFrame(results)
    
    def get_targets(self) -> pd.DataFrame:
        """
        Get targets from database.
        
        Returns:
            DataFrame containing target data
        """
        query = "SELECT * FROM targets"
        results = self.execute_query(query)
        return pd.DataFrame(results)
    
    def get_activities(self, molecule_id: Optional[int] = None, 
                      target_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get activities from database.
        
        Args:
            molecule_id: Filter by molecule ID
            target_id: Filter by target ID
            
        Returns:
            DataFrame containing activity data
        """
        query = "SELECT * FROM activities WHERE 1=1"
        params = []
        
        if molecule_id:
            query += " AND molecule_id = ?"
            params.append(molecule_id)
        
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        
        results = self.execute_query(query, tuple(params))
        return pd.DataFrame(results)
    
    def get_predictions(self, molecule_id: Optional[int] = None,
                       target_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get predictions from database.
        
        Args:
            molecule_id: Filter by molecule ID
            target_id: Filter by target ID
            
        Returns:
            DataFrame containing prediction data
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if molecule_id:
            query += " AND molecule_id = ?"
            params.append(molecule_id)
        
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        
        results = self.execute_query(query, tuple(params))
        return pd.DataFrame(results)
    
    def load_chembl_data(self, target_ids: List[str], limit: Optional[int] = None):
        """
        Load ChEMBL data into database.
        
        Args:
            target_ids: List of ChEMBL target IDs
            limit: Maximum number of compounds to load
        """
        self.logger.info(f"Loading ChEMBL data for {len(target_ids)} targets")
        
        # Create tables if they don't exist
        self.create_tables()
        
        # Load data for each target
        for target_id in target_ids:
            try:
                self.logger.info(f"Loading data for target: {target_id}")
                
                # Load target information
                target_info = self.chembl_loader.get_target_info(target_id)
                if target_info:
                    target_db_id = self.insert_target(target_info)
                    self.logger.info(f"Inserted target: {target_info['target_name']}")
                
                # Load activities
                activities = self.chembl_loader.get_activities(target_id, limit=limit)
                for activity in activities:
                    # Insert molecule if not exists
                    molecule_query = "SELECT id FROM molecules WHERE chembl_id = ?"
                    molecule_result = self.execute_query(molecule_query, (activity['molecule_chembl_id'],))
                    
                    if not molecule_result:
                        molecule_data = {
                            'chembl_id': activity['molecule_chembl_id'],
                            'smiles': activity.get('canonical_smiles'),
                            'mol_weight': activity.get('molecular_weight'),
                            'logp': activity.get('alogp'),
                            'hbd': activity.get('hbd'),
                            'hba': activity.get('hba'),
                            'tpsa': activity.get('tpsa'),
                            'rotatable_bonds': activity.get('rotatable_bonds'),
                            'aromatic_rings': activity.get('aromatic_rings'),
                            'heavy_atoms': activity.get('heavy_atoms')
                        }
                        molecule_id = self.insert_molecule(molecule_data)
                    else:
                        molecule_id = molecule_result[0]['id']
                    
                    # Insert activity
                    activity_data = {
                        'molecule_id': molecule_id,
                        'target_id': target_db_id,
                        'activity_type': activity.get('standard_type'),
                        'activity_value': activity.get('standard_value'),
                        'activity_unit': activity.get('standard_units'),
                        'activity_relation': activity.get('standard_relation')
                    }
                    self.insert_activity(activity_data)
                
                self.logger.info(f"Loaded {len(activities)} activities for target {target_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load data for target {target_id}: {e}")
                continue
        
        self.logger.info("ChEMBL data loading completed")


class LocalDatabase(DatabaseManager):
    """Local database manager for custom data."""
    
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize local database manager.
        
        Args:
            db_path: Path to local database file
            logger: Logger instance
        """
        super().__init__(db_path, logger)
        
    def create_tables(self):
        """Create local database tables."""
        create_queries = [
            """
            CREATE TABLE IF NOT EXISTS custom_molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id TEXT UNIQUE,
                smiles TEXT,
                features TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS custom_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id TEXT,
                target_name TEXT,
                prediction_value REAL,
                confidence REAL,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_version TEXT,
                checkpoint_path TEXT,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for query in create_queries:
            self.execute_update(query)
        
        self.logger.info("Local database tables created successfully")
    
    def insert_custom_molecule(self, molecule_data: Dict[str, Any]) -> int:
        """
        Insert custom molecule data.
        
        Args:
            molecule_data: Dictionary containing molecule information
            
        Returns:
            Inserted molecule ID
        """
        query = """
        INSERT INTO custom_molecules (molecule_id, smiles, features, properties)
        VALUES (?, ?, ?, ?)
        """
        
        params = (
            molecule_data.get('molecule_id'),
            molecule_data.get('smiles'),
            json.dumps(molecule_data.get('features', {})),
            json.dumps(molecule_data.get('properties', {}))
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def insert_custom_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """
        Insert custom prediction data.
        
        Args:
            prediction_data: Dictionary containing prediction information
            
        Returns:
            Inserted prediction ID
        """
        query = """
        INSERT INTO custom_predictions (molecule_id, target_name, prediction_value, 
                                       confidence, model_name)
        VALUES (?, ?, ?, ?, ?)
        """
        
        params = (
            prediction_data.get('molecule_id'),
            prediction_data.get('target_name'),
            prediction_data.get('prediction_value'),
            prediction_data.get('confidence'),
            prediction_data.get('model_name')
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def insert_model_checkpoint(self, checkpoint_data: Dict[str, Any]) -> int:
        """
        Insert model checkpoint data.
        
        Args:
            checkpoint_data: Dictionary containing checkpoint information
            
        Returns:
            Inserted checkpoint ID
        """
        query = """
        INSERT INTO model_checkpoints (model_name, model_version, checkpoint_path, metrics)
        VALUES (?, ?, ?, ?)
        """
        
        params = (
            checkpoint_data.get('model_name'),
            checkpoint_data.get('model_version'),
            checkpoint_data.get('checkpoint_path'),
            json.dumps(checkpoint_data.get('metrics', {}))
        )
        
        self.execute_update(query, params)
        
        # Get inserted ID
        result = self.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id']
    
    def get_custom_molecules(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get custom molecules from database.
        
        Args:
            limit: Maximum number of molecules to return
            
        Returns:
            DataFrame containing custom molecule data
        """
        query = "SELECT * FROM custom_molecules"
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.execute_query(query)
        df = pd.DataFrame(results)
        
        # Parse JSON fields
        if not df.empty:
            df['features'] = df['features'].apply(lambda x: json.loads(x) if x else {})
            df['properties'] = df['properties'].apply(lambda x: json.loads(x) if x else {})
        
        return df
    
    def get_custom_predictions(self, molecule_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get custom predictions from database.
        
        Args:
            molecule_id: Filter by molecule ID
            
        Returns:
            DataFrame containing custom prediction data
        """
        query = "SELECT * FROM custom_predictions WHERE 1=1"
        params = []
        
        if molecule_id:
            query += " AND molecule_id = ?"
            params.append(molecule_id)
        
        results = self.execute_query(query, tuple(params))
        return pd.DataFrame(results)
    
    def get_model_checkpoints(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get model checkpoints from database.
        
        Args:
            model_name: Filter by model name
            
        Returns:
            DataFrame containing model checkpoint data
        """
        query = "SELECT * FROM model_checkpoints WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        results = self.execute_query(query, tuple(params))
        df = pd.DataFrame(results)
        
        # Parse JSON fields
        if not df.empty:
            df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else {})
        
        return df
