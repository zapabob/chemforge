"""
全テストの実行スクリプト

ChemForgeの全モジュールのテストを実行
"""

import unittest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """全テストを実行"""
    # テストモジュールのリスト
    test_modules = [
        'tests.test_molecular_features',
        'tests.test_cns_scaffolds',
        'tests.test_property_predictor',
        'tests.test_scaffold_generator',
        'tests.test_gui_widgets'
    ]
    
    # テストスイートを作成
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            tests = unittest.TestLoader().loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✅ {module_name} のテストを追加")
        except ImportError as e:
            print(f"❌ {module_name} のインポートに失敗: {e}")
        except Exception as e:
            print(f"❌ {module_name} のテストロードに失敗: {e}")
    
    # テストを実行
    print("\n" + "="*50)
    print("🧪 ChemForge テストスイート実行開始")
    print("="*50)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    print("\n" + "="*50)
    print("📊 テスト結果サマリー")
    print("="*50)
    print(f"実行したテスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ エラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 全テストが成功しました！")
        return True
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
