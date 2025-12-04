# no-rank-tensor-decomposition/metric_learning/__init__.py
"""
No-rank Tensor Decomposition Using Metric Learning
"""

from .models import FaceMetricLearningModel
from .datasets import EnhancedFaceDataset, OlivettiFaceDataset
from .losses import ImprovedFaceMetricLoss
from .decomposition import ReconstructionAnalyzer, cp_decomposition_embeddings, tucker_decomposition_embeddings
from .metrics import calculate_all_metrics, calculate_separation_ratio, calculate_clustering_metrics
from .visualization import (
    visualize_tensor_comparison_with_reconstruction,
    visualize_tensor_decomposition_comparison,
    show_sample_reconstructions_separate
)
from .utils import (
    improved_smart_triplet_mining,
    get_simple_triplets,
    evaluate_face_embeddings,
    add_regularization_to_training
)
from .main import compare_tensor_decomposition_methods, main

__version__ = "0.1.0"
__author__ = "Maryam Bagherian"

__all__ = [
    'FaceMetricLearningModel',
    'EnhancedFaceDataset',
    'OlivettiFaceDataset',
    'ImprovedFaceMetricLoss',
    'ReconstructionAnalyzer',
    'cp_decomposition_embeddings',
    'tucker_decomposition_embeddings',
    'calculate_all_metrics',
    'calculate_separation_ratio',
    'calculate_clustering_metrics',
    'visualize_tensor_comparison_with_reconstruction',
    'visualize_tensor_decomposition_comparison',
    'improved_smart_triplet_mining',
    'get_simple_triplets',
    'evaluate_face_embeddings',
    'add_regularization_to_training',
    'compare_tensor_decomposition_methods',
    'main'
]