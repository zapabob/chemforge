"""
File utilities for ChemForge platform.

This module provides file management functionality including data import/export,
file operations, and data persistence.
"""

import pandas as pd
import numpy as np
import json
import pickle
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import h5py
import sqlite3
from datetime import datetime
import gzip
import shutil


class FileManager:
    """File management utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize file manager.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists.
        
        Args:
            directory: Directory path
            
        Returns:
            Path object
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: File path
            
        Returns:
            Dictionary containing file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.suffix,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir()
        }
    
    def list_files(self, directory: Union[str, Path], 
                   pattern: str = "*",
                   recursive: bool = False) -> List[Path]:
        """
        List files in directory.
        
        Args:
            directory: Directory path
            pattern: File pattern
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    def copy_file(self, source: Union[str, Path], 
                  destination: Union[str, Path],
                  overwrite: bool = False) -> Path:
        """
        Copy file.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            Destination path
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination file exists: {destination}")
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source, destination)
        self.logger.info(f"Copied file: {source} -> {destination}")
        
        return destination
    
    def move_file(self, source: Union[str, Path], 
                  destination: Union[str, Path],
                  overwrite: bool = False) -> Path:
        """
        Move file.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            Destination path
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination file exists: {destination}")
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(source), str(destination))
        self.logger.info(f"Moved file: {source} -> {destination}")
        
        return destination
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete file.
        
        Args:
            file_path: File path
            
        Returns:
            True if successful
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return False
        
        file_path.unlink()
        self.logger.info(f"Deleted file: {file_path}")
        return True
    
    def compress_file(self, file_path: Union[str, Path],
                     compression: str = 'gzip') -> Path:
        """
        Compress file.
        
        Args:
            file_path: File path
            compression: Compression type
            
        Returns:
            Compressed file path
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if compression == 'gzip':
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
        
        self.logger.info(f"Compressed file: {file_path} -> {compressed_path}")
        return compressed_path
    
    def decompress_file(self, file_path: Union[str, Path]) -> Path:
        """
        Decompress file.
        
        Args:
            file_path: Compressed file path
            
        Returns:
            Decompressed file path
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.gz':
            decompressed_path = file_path.with_suffix('')
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f"File is not compressed: {file_path}")
        
        self.logger.info(f"Decompressed file: {file_path} -> {decompressed_path}")
        return decompressed_path


class DataExporter:
    """Data export utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data exporter.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def export_csv(self, data: pd.DataFrame, 
                   file_path: Union[str, Path],
                   index: bool = False) -> Path:
        """
        Export data to CSV.
        
        Args:
            data: DataFrame to export
            file_path: Output file path
            index: Whether to include index
            
        Returns:
            Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(file_path, index=index)
        self.logger.info(f"Exported CSV: {file_path}")
        
        return file_path
    
    def export_json(self, data: Union[Dict, List], 
                    file_path: Union[str, Path],
                    indent: int = 2) -> Path:
        """
        Export data to JSON.
        
        Args:
            data: Data to export
            file_path: Output file path
            indent: JSON indentation
            
        Returns:
            Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        self.logger.info(f"Exported JSON: {file_path}")
        return file_path
    
    def export_pickle(self, data: Any, 
                      file_path: Union[str, Path]) -> Path:
        """
        Export data to pickle.
        
        Args:
            data: Data to export
            file_path: Output file path
            
        Returns:
            Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Exported pickle: {file_path}")
        return file_path
    
    def export_hdf5(self, data: Dict[str, Any], 
                    file_path: Union[str, Path]) -> Path:
        """
        Export data to HDF5.
        
        Args:
            data: Dictionary containing data arrays
            file_path: Output file path
            
        Returns:
            Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(file_path, 'w') as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, pd.DataFrame):
                    f.create_dataset(key, data=value.values)
                else:
                    f.attrs[key] = value
        
        self.logger.info(f"Exported HDF5: {file_path}")
        return file_path
    
    def export_sqlite(self, data: pd.DataFrame, 
                      file_path: Union[str, Path],
                      table_name: str = 'data') -> Path:
        """
        Export data to SQLite.
        
        Args:
            data: DataFrame to export
            file_path: Output file path
            table_name: Table name
            
        Returns:
            Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(file_path) as conn:
            data.to_sql(table_name, conn, if_exists='replace', index=False)
        
        self.logger.info(f"Exported SQLite: {file_path}")
        return file_path
    
    def export_excel(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     file_path: Union[str, Path],
                     sheet_name: str = 'Sheet1') -> Path:
        """
        Export data to Excel.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            file_path: Output file path
            sheet_name: Sheet name (for single DataFrame)
            
        Returns:
            Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_excel(file_path, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(file_path) as writer:
                for sheet, df in data.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)
        
        self.logger.info(f"Exported Excel: {file_path}")
        return file_path


class DataImporter:
    """Data import utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data importer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def import_csv(self, file_path: Union[str, Path], 
                   **kwargs) -> pd.DataFrame:
        """
        Import data from CSV.
        
        Args:
            file_path: Input file path
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_csv(file_path, **kwargs)
        self.logger.info(f"Imported CSV: {file_path} ({len(data)} rows)")
        
        return data
    
    def import_json(self, file_path: Union[str, Path]) -> Union[Dict, List]:
        """
        Import data from JSON.
        
        Args:
            file_path: Input file path
            
        Returns:
            Data from JSON
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Imported JSON: {file_path}")
        return data
    
    def import_pickle(self, file_path: Union[str, Path]) -> Any:
        """
        Import data from pickle.
        
        Args:
            file_path: Input file path
            
        Returns:
            Data from pickle
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.logger.info(f"Imported pickle: {file_path}")
        return data
    
    def import_hdf5(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Import data from HDF5.
        
        Args:
            file_path: Input file path
            
        Returns:
            Dictionary containing data arrays
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
            
            # Import attributes
            for key, value in f.attrs.items():
                data[f"attr_{key}"] = value
        
        self.logger.info(f"Imported HDF5: {file_path}")
        return data
    
    def import_sqlite(self, file_path: Union[str, Path], 
                      table_name: str = 'data') -> pd.DataFrame:
        """
        Import data from SQLite.
        
        Args:
            file_path: Input file path
            table_name: Table name
            
        Returns:
            DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with sqlite3.connect(file_path) as conn:
            data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        self.logger.info(f"Imported SQLite: {file_path} ({len(data)} rows)")
        return data
    
    def import_excel(self, file_path: Union[str, Path], 
                     sheet_name: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Import data from Excel.
        
        Args:
            file_path: Input file path
            sheet_name: Sheet name (None for all sheets)
            
        Returns:
            DataFrame or dictionary of DataFrames
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if sheet_name:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            self.logger.info(f"Imported Excel sheet '{sheet_name}': {file_path}")
        else:
            data = pd.read_excel(file_path, sheet_name=None)
            self.logger.info(f"Imported Excel all sheets: {file_path}")
        
        return data
    
    def import_chembl_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Import ChEMBL data from CSV.
        
        Args:
            file_path: Input file path
            
        Returns:
            DataFrame with ChEMBL data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read CSV with ChEMBL-specific parameters
        data = pd.read_csv(
            file_path,
            sep=',',
            low_memory=False,
            na_values=['', 'NULL', 'null', 'NaN', 'nan']
        )
        
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Convert numeric columns
        numeric_columns = ['molecular_weight', 'alogp', 'hbd', 'hba', 'tpsa', 
                         'rotatable_bonds', 'aromatic_rings', 'heavy_atoms']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        self.logger.info(f"Imported ChEMBL data: {file_path} ({len(data)} rows)")
        return data
    
    def import_molecular_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Import molecular data from CSV.
        
        Args:
            file_path: Input file path
            
        Returns:
            DataFrame with molecular data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read CSV with molecular-specific parameters
        data = pd.read_csv(
            file_path,
            sep=',',
            low_memory=False,
            na_values=['', 'NULL', 'null', 'NaN', 'nan']
        )
        
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Convert numeric columns
        numeric_columns = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa', 
                         'rotatable_bonds', 'aromatic_rings', 'heavy_atoms']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        self.logger.info(f"Imported molecular data: {file_path} ({len(data)} rows)")
        return data
