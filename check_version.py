import sys
import site
import os

def run_version_check():
    # Check current working directory
    print("Current Working Directory:", os.getcwd())
    
    # Check Python version
    print("Python version:", sys.version)
    #print("Python executable:", sys.executable)
    
    # Check the site-packages directories
    #print("Site-packages locations:", site.getsitepackages())
    
    # Print sys.path
    #print("Python Path:", sys.path)

    # Check Haystack version
    try:
        print("Trying to import Haystack...")
        from haystack import __version__ as haystack_version  # Correct import statement
        print("Haystack version:", haystack_version)
    except ImportError as e:
        print("Haystack is not installed.", e)

if __name__ == '__main__':
    run_version_check()
