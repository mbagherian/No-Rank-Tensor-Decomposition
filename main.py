# no-rank-tensor-decomposition/main.py
#!/usr/bin/env python3
"""
Entry point for the No-rank Tensor Decomposition package.
Run with: python main.py
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metric_learning import main

if __name__ == "__main__":
    print("="*70)
    print("NO-RANK TENSOR DECOMPOSITION USING METRIC LEARNING")
    print("="*70)
    print("Implementation of the paper by Maryam Bagherian")
    print("arXiv: https://arxiv.org/abs/2511.01816")
    print("="*70)
    
    main()