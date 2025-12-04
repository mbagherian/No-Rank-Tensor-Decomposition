# no-rank-tensor-decomposition/examples/compare_methods.py
#!/usr/bin/env python3
"""
Script to run both datasets and compare results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from examples.run_lfw import main as run_lfw
from examples.run_olivetti import main as run_olivetti

def compare_datasets():
    """Compare results from both datasets"""
    print("📊 COMPARING DATASET PERFORMANCE")
    print("="*60)
    
    results = []
    
    # Run LFW experiment
    print("\n1️⃣ Running LFW Dataset...")
    # In a real implementation, you would capture the results from run_lfw
    # For now, we'll simulate or you can modify to capture actual results
    
    # Run Olivetti experiment
    print("\n2️⃣ Running Olivetti Dataset...")
    # Similarly capture results from run_olivetti
    
    print("\n📈 Comparison completed!")
    
    # This is a placeholder - you would need to modify the example scripts
    # to return results that can be compared here

if __name__ == "__main__":
    compare_datasets()