"""
Unit tests for file utilities.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import json
import pickle
import h5py
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from chemforge.utils.file_utils import FileManager, DataExporter, DataImporter


class TestFileManager(unittest.TestCase):
    """Test FileManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = FileManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test FileManager initialization."""
        self.assertIsNotNone(self.file_manager.logger)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        test_dir = Path(self.temp_dir) / "test_dir"
        result = self.file_manager.ensure_directory(test_dir)
        self.assertEqual(result, test_dir)
        self.assertTrue(test_dir.exists())
        self.assertTrue(test_dir.is_dir())
    
    def test_ensure_directory_nested(self):
        """Test nested directory creation."""
        nested_dir = Path(self.temp_dir) / "level1" / "level2" / "level3"
        result = self.file_manager.ensure_directory(nested_dir)
        self.assertEqual(result, nested_dir)
        self.assertTrue(nested_dir.exists())
        self.assertTrue(nested_dir.is_dir())
    
    def test_get_file_info(self):
        """Test file information retrieval."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        info = self.file_manager.get_file_info(test_file)
        
        self.assertEqual(info['name'], 'test.txt')
        self.assertEqual(info['path'], str(test_file))
        self.assertGreater(info['size'], 0)
        self.assertIsNotNone(info['created'])
        self.assertIsNotNone(info['modified'])
        self.assertEqual(info['extension'], '.txt')
        self.assertTrue(info['is_file'])
        self.assertFalse(info['is_dir'])
    
    def test_get_file_info_nonexistent(self):
        """Test file information retrieval for nonexistent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        with self.assertRaises(FileNotFoundError):
            self.file_manager.get_file_info(nonexistent_file)
    
    def test_list_files(self):
        """Test file listing."""
        # Create test files
        test_files = ['file1.txt', 'file2.txt', 'file3.csv']
        for filename in test_files:
            (Path(self.temp_dir) / filename).write_text("test content")
        
        # List all files
        files = self.file_manager.list_files(self.temp_dir)
        file_names = [f.name for f in files]
        
        for filename in test_files:
            self.assertIn(filename, file_names)
    
    def test_list_files_pattern(self):
        """Test file listing with pattern."""
        # Create test files
        test_files = ['file1.txt', 'file2.txt', 'file3.csv']
        for filename in test_files:
            (Path(self.temp_dir) / filename).write_text("test content")
        
        # List only .txt files
        txt_files = self.file_manager.list_files(self.temp_dir, pattern="*.txt")
        txt_file_names = [f.name for f in txt_files]
        
        self.assertEqual(len(txt_files), 2)
        self.assertIn('file1.txt', txt_file_names)
        self.assertIn('file2.txt', txt_file_names)
        self.assertNotIn('file3.csv', txt_file_names)
    
    def test_list_files_recursive(self):
        """Test recursive file listing."""
        # Create nested structure
        subdir = Path(self.temp_dir) / "subdir"
        subdir.mkdir()
        
        # Create files in root and subdir
        (Path(self.temp_dir) / "root.txt").write_text("test")
        (subdir / "sub.txt").write_text("test")
        
        # List files recursively
        files = self.file_manager.list_files(self.temp_dir, recursive=True)
        file_names = [f.name for f in files]
        
        self.assertIn('root.txt', file_names)
        self.assertIn('sub.txt', file_names)
    
    def test_copy_file(self):
        """Test file copying."""
        # Create source file
        source_file = Path(self.temp_dir) / "source.txt"
        source_file.write_text("test content")
        
        # Copy file
        dest_file = Path(self.temp_dir) / "dest.txt"
        result = self.file_manager.copy_file(source_file, dest_file)
        
        self.assertEqual(result, dest_file)
        self.assertTrue(dest_file.exists())
        self.assertEqual(dest_file.read_text(), "test content")
    
    def test_copy_file_overwrite(self):
        """Test file copying with overwrite."""
        # Create source file
        source_file = Path(self.temp_dir) / "source.txt"
        source_file.write_text("test content")
        
        # Create existing destination file
        dest_file = Path(self.temp_dir) / "dest.txt"
        dest_file.write_text("old content")
        
        # Copy file with overwrite
        result = self.file_manager.copy_file(source_file, dest_file, overwrite=True)
        
        self.assertEqual(result, dest_file)
        self.assertEqual(dest_file.read_text(), "test content")
    
    def test_copy_file_no_overwrite(self):
        """Test file copying without overwrite."""
        # Create source file
        source_file = Path(self.temp_dir) / "source.txt"
        source_file.write_text("test content")
        
        # Create existing destination file
        dest_file = Path(self.temp_dir) / "dest.txt"
        dest_file.write_text("old content")
        
        # Copy file without overwrite
        with self.assertRaises(FileExistsError):
            self.file_manager.copy_file(source_file, dest_file, overwrite=False)
    
    def test_move_file(self):
        """Test file moving."""
        # Create source file
        source_file = Path(self.temp_dir) / "source.txt"
        source_file.write_text("test content")
        
        # Move file
        dest_file = Path(self.temp_dir) / "dest.txt"
        result = self.file_manager.move_file(source_file, dest_file)
        
        self.assertEqual(result, dest_file)
        self.assertTrue(dest_file.exists())
        self.assertFalse(source_file.exists())
        self.assertEqual(dest_file.read_text(), "test content")
    
    def test_delete_file(self):
        """Test file deletion."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Delete file
        result = self.file_manager.delete_file(test_file)
        
        self.assertTrue(result)
        self.assertFalse(test_file.exists())
    
    def test_delete_file_nonexistent(self):
        """Test deletion of nonexistent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        result = self.file_manager.delete_file(nonexistent_file)
        
        self.assertFalse(result)
    
    def test_compress_file(self):
        """Test file compression."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Compress file
        compressed_file = self.file_manager.compress_file(test_file)
        
        self.assertEqual(compressed_file.suffix, '.gz')
        self.assertTrue(compressed_file.exists())
        self.assertTrue(compressed_file.stat().st_size < test_file.stat().st_size)
    
    def test_decompress_file(self):
        """Test file decompression."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Compress file
        compressed_file = self.file_manager.compress_file(test_file)
        
        # Decompress file
        decompressed_file = self.file_manager.decompress_file(compressed_file)
        
        self.assertEqual(decompressed_file.suffix, '')
        self.assertTrue(decompressed_file.exists())
        self.assertEqual(decompressed_file.read_text(), "test content")


class TestDataExporter(unittest.TestCase):
    """Test DataExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [1.5, 2.5, 3.5]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test DataExporter initialization."""
        self.assertIsNotNone(self.exporter.logger)
    
    def test_export_csv(self):
        """Test CSV export."""
        output_path = Path(self.temp_dir) / "test.csv"
        result = self.exporter.export_csv(self.test_data, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        loaded_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_export_csv_with_index(self):
        """Test CSV export with index."""
        output_path = Path(self.temp_dir) / "test_with_index.csv"
        result = self.exporter.export_csv(self.test_data, output_path, index=True)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
    
    def test_export_json(self):
        """Test JSON export."""
        test_dict = {'key1': 'value1', 'key2': [1, 2, 3]}
        output_path = Path(self.temp_dir) / "test.json"
        result = self.exporter.export_json(test_dict, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, test_dict)
    
    def test_export_pickle(self):
        """Test pickle export."""
        test_obj = {'key1': 'value1', 'key2': [1, 2, 3]}
        output_path = Path(self.temp_dir) / "test.pkl"
        result = self.exporter.export_pickle(test_obj, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.assertEqual(loaded_data, test_obj)
    
    def test_export_hdf5(self):
        """Test HDF5 export."""
        test_data = {
            'array1': np.array([1, 2, 3]),
            'array2': np.array([[1, 2], [3, 4]]),
            'dataframe': self.test_data
        }
        output_path = Path(self.temp_dir) / "test.h5"
        result = self.exporter.export_hdf5(test_data, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with h5py.File(output_path, 'r') as f:
            self.assertIn('array1', f)
            self.assertIn('array2', f)
            self.assertIn('dataframe', f)
    
    def test_export_sqlite(self):
        """Test SQLite export."""
        output_path = Path(self.temp_dir) / "test.db"
        result = self.exporter.export_sqlite(self.test_data, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with sqlite3.connect(output_path) as conn:
            loaded_data = pd.read_sql_query("SELECT * FROM data", conn)
            pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_export_excel_single_sheet(self):
        """Test Excel export with single sheet."""
        output_path = Path(self.temp_dir) / "test.xlsx"
        result = self.exporter.export_excel(self.test_data, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        loaded_data = pd.read_excel(output_path)
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_export_excel_multiple_sheets(self):
        """Test Excel export with multiple sheets."""
        data_dict = {
            'sheet1': self.test_data,
            'sheet2': self.test_data.copy()
        }
        output_path = Path(self.temp_dir) / "test_multi.xlsx"
        result = self.exporter.export_excel(data_dict, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with pd.ExcelFile(output_path) as xls:
            self.assertIn('sheet1', xls.sheet_names)
            self.assertIn('sheet2', xls.sheet_names)


class TestDataImporter(unittest.TestCase):
    """Test DataImporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.importer = DataImporter()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [1.5, 2.5, 3.5]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test DataImporter initialization."""
        self.assertIsNotNone(self.importer.logger)
    
    def test_import_csv(self):
        """Test CSV import."""
        # Create test CSV file
        csv_path = Path(self.temp_dir) / "test.csv"
        self.test_data.to_csv(csv_path, index=False)
        
        # Import CSV
        loaded_data = self.importer.import_csv(csv_path)
        
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_import_csv_with_kwargs(self):
        """Test CSV import with additional arguments."""
        # Create test CSV file with custom separator
        csv_path = Path(self.temp_dir) / "test.csv"
        self.test_data.to_csv(csv_path, index=False, sep=';')
        
        # Import CSV with separator
        loaded_data = self.importer.import_csv(csv_path, sep=';')
        
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_import_csv_nonexistent(self):
        """Test CSV import with nonexistent file."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.csv"
        with self.assertRaises(FileNotFoundError):
            self.importer.import_csv(nonexistent_path)
    
    def test_import_json(self):
        """Test JSON import."""
        test_dict = {'key1': 'value1', 'key2': [1, 2, 3]}
        json_path = Path(self.temp_dir) / "test.json"
        
        with open(json_path, 'w') as f:
            json.dump(test_dict, f)
        
        # Import JSON
        loaded_data = self.importer.import_json(json_path)
        
        self.assertEqual(loaded_data, test_dict)
    
    def test_import_pickle(self):
        """Test pickle import."""
        test_obj = {'key1': 'value1', 'key2': [1, 2, 3]}
        pickle_path = Path(self.temp_dir) / "test.pkl"
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(test_obj, f)
        
        # Import pickle
        loaded_data = self.importer.import_pickle(pickle_path)
        
        self.assertEqual(loaded_data, test_obj)
    
    def test_import_hdf5(self):
        """Test HDF5 import."""
        hdf5_path = Path(self.temp_dir) / "test.h5"
        
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('array1', data=np.array([1, 2, 3]))
            f.create_dataset('array2', data=np.array([[1, 2], [3, 4]]))
            f.attrs['attr1'] = 'value1'
        
        # Import HDF5
        loaded_data = self.importer.import_hdf5(hdf5_path)
        
        self.assertIn('array1', loaded_data)
        self.assertIn('array2', loaded_data)
        self.assertIn('attr_attr1', loaded_data)
        np.testing.assert_array_equal(loaded_data['array1'], np.array([1, 2, 3]))
    
    def test_import_sqlite(self):
        """Test SQLite import."""
        sqlite_path = Path(self.temp_dir) / "test.db"
        
        with sqlite3.connect(sqlite_path) as conn:
            self.test_data.to_sql('data', conn, index=False)
        
        # Import SQLite
        loaded_data = self.importer.import_sqlite(sqlite_path)
        
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_import_excel_single_sheet(self):
        """Test Excel import with single sheet."""
        excel_path = Path(self.temp_dir) / "test.xlsx"
        self.test_data.to_excel(excel_path, index=False)
        
        # Import Excel
        loaded_data = self.importer.import_excel(excel_path, sheet_name='Sheet1')
        
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_import_excel_all_sheets(self):
        """Test Excel import with all sheets."""
        excel_path = Path(self.temp_dir) / "test.xlsx"
        
        with pd.ExcelWriter(excel_path) as writer:
            self.test_data.to_excel(writer, sheet_name='sheet1', index=False)
            self.test_data.to_excel(writer, sheet_name='sheet2', index=False)
        
        # Import Excel
        loaded_data = self.importer.import_excel(excel_path)
        
        self.assertIsInstance(loaded_data, dict)
        self.assertIn('sheet1', loaded_data)
        self.assertIn('sheet2', loaded_data)
        pd.testing.assert_frame_equal(loaded_data['sheet1'], self.test_data)
    
    def test_import_chembl_data(self):
        """Test ChEMBL data import."""
        # Create test ChEMBL CSV file
        chembl_data = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'canonical_smiles': ['CCO', 'CCN', 'CC(C)O'],
            'molecular_weight': [46.07, 45.08, 60.10],
            'alogp': [0.31, 0.16, 0.05],
            'hbd': [1, 2, 1],
            'hba': [1, 1, 1],
            'tpsa': [20.23, 26.02, 20.23],
            'rotatable_bonds': [0, 0, 0],
            'aromatic_rings': [0, 0, 0],
            'heavy_atoms': [2, 2, 3]
        })
        
        csv_path = Path(self.temp_dir) / "chembl_test.csv"
        chembl_data.to_csv(csv_path, index=False)
        
        # Import ChEMBL data
        loaded_data = self.importer.import_chembl_data(csv_path)
        
        self.assertEqual(len(loaded_data), 3)
        self.assertIn('molecule_chembl_id', loaded_data.columns)
        self.assertIn('canonical_smiles', loaded_data.columns)
    
    def test_import_molecular_data(self):
        """Test molecular data import."""
        # Create test molecular CSV file
        molecular_data = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CC(C)O'],
            'mol_weight': [46.07, 45.08, 60.10],
            'logp': [0.31, 0.16, 0.05],
            'hbd': [1, 2, 1],
            'hba': [1, 1, 1],
            'tpsa': [20.23, 26.02, 20.23],
            'rotatable_bonds': [0, 0, 0],
            'aromatic_rings': [0, 0, 0],
            'heavy_atoms': [2, 2, 3]
        })
        
        csv_path = Path(self.temp_dir) / "molecular_test.csv"
        molecular_data.to_csv(csv_path, index=False)
        
        # Import molecular data
        loaded_data = self.importer.import_molecular_data(csv_path)
        
        self.assertEqual(len(loaded_data), 3)
        self.assertIn('smiles', loaded_data.columns)
        self.assertIn('mol_weight', loaded_data.columns)


if __name__ == '__main__':
    unittest.main()
