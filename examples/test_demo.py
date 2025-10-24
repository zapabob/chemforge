"""
ChemForge Test Demo

This module demonstrates running tests for ChemForge components.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

from chemforge.utils.logging_utils import setup_logging


def run_unit_tests():
    """Run unit tests."""
    print("Running Unit Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info("Running ChemForge Unit Tests")
    
    try:
        # Run unit tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
            print(result.stdout)
        else:
            print("❌ Unit tests failed")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running unit tests: {e}")
        return False


def run_integration_tests():
    """Run integration tests."""
    print("Running Integration Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info("Running ChemForge Integration Tests")
    
    try:
        # Run integration tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_integration.py", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Integration tests passed")
            print(result.stdout)
        else:
            print("❌ Integration tests failed")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return False


def run_performance_tests():
    """Run performance tests."""
    print("Running Performance Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info("Running ChemForge Performance Tests")
    
    try:
        # Run performance tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_performance.py", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Performance tests passed")
            print(result.stdout)
        else:
            print("❌ Performance tests failed")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running performance tests: {e}")
        return False


def run_coverage_tests():
    """Run coverage tests."""
    print("Running Coverage Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info("Running ChemForge Coverage Tests")
    
    try:
        # Run coverage tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "--cov=chemforge", "--cov-report=html", "--cov-report=term", "tests/"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Coverage tests passed")
            print(result.stdout)
        else:
            print("❌ Coverage tests failed")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running coverage tests: {e}")
        return False


def run_specific_tests(test_pattern):
    """Run specific tests matching pattern."""
    print(f"Running Tests Matching Pattern: {test_pattern}")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info(f"Running ChemForge Tests Matching: {test_pattern}")
    
    try:
        # Run specific tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-k", test_pattern, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Tests matching '{test_pattern}' passed")
            print(result.stdout)
        else:
            print(f"❌ Tests matching '{test_pattern}' failed")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running specific tests: {e}")
        return False


def run_benchmark_tests():
    """Run benchmark tests."""
    print("Running Benchmark Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info("Running ChemForge Benchmark Tests")
    
    try:
        # Run benchmark tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_performance.py::TestChemForgePerformance::test_benchmark_comparison", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Benchmark tests passed")
            print(result.stdout)
        else:
            print("❌ Benchmark tests failed")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running benchmark tests: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("Running All Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('test_demo')
    logger.info("Running All ChemForge Tests")
    
    results = {}
    
    # Run unit tests
    print("\n1. Unit Tests")
    results['unit'] = run_unit_tests()
    
    # Run integration tests
    print("\n2. Integration Tests")
    results['integration'] = run_integration_tests()
    
    # Run performance tests
    print("\n3. Performance Tests")
    results['performance'] = run_performance_tests()
    
    # Run coverage tests
    print("\n4. Coverage Tests")
    results['coverage'] = run_coverage_tests()
    
    # Summary
    print("\nTest Results Summary:")
    print("=" * 50)
    
    for test_type, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_type.upper()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


def run_test_suite():
    """Run comprehensive test suite."""
    print("ChemForge Test Suite")
    print("=" * 50)
    print()
    print("Available test options:")
    print("1. Unit Tests")
    print("2. Integration Tests")
    print("3. Performance Tests")
    print("4. Coverage Tests")
    print("5. Specific Tests")
    print("6. Benchmark Tests")
    print("7. All Tests")
    print()
    
    choice = input("Enter your choice (1-7): ").strip()
    
    if choice == "1":
        return run_unit_tests()
    elif choice == "2":
        return run_integration_tests()
    elif choice == "3":
        return run_performance_tests()
    elif choice == "4":
        return run_coverage_tests()
    elif choice == "5":
        pattern = input("Enter test pattern: ").strip()
        return run_specific_tests(pattern)
    elif choice == "6":
        return run_benchmark_tests()
    elif choice == "7":
        return run_all_tests()
    else:
        print("Invalid choice. Please run the test suite again.")
        return False


def main():
    """Main test demo function."""
    print("ChemForge Test Demo")
    print("=" * 50)
    print()
    print("This demo runs various tests for ChemForge components.")
    print()
    
    # Check if pytest is available
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return
    
    # Check if coverage is available
    try:
        import coverage
        print("✅ coverage is available")
    except ImportError:
        print("⚠️  coverage not found. Install with: pip install coverage")
    
    # Run test suite
    success = run_test_suite()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")


if __name__ == "__main__":
    main()
