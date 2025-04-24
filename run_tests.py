import unittest
import sys
import os
import coverage

def run_tests():
    # Start code coverage
    cov = coverage.Coverage(branch=True, source=['data_fetchers', 'ml_backtesting', 'scoring', 'main', 'visualizations', 'sheets_handler', 'model_retrainer'])
    cov.start()

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('.')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Stop code coverage
    cov.stop()
    cov.save()

    # Print coverage report
    print("\nCode Coverage:")
    cov.report()

    # Generate HTML coverage report
    cov.html_report(directory='htmlcov')
    print("HTML coverage report generated in 'htmlcov' directory")

    return result

if __name__ == '__main__':
    result = run_tests()
    
    if result.wasSuccessful():
        print("All tests passed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)