# No-rank Tensor Decomposition Using Metric Learning
https://img.shields.io/badge/arXiv-2511.01816-b31b1b.svg
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/PyTorch-%2523EE4C2C.svg?logo=PyTorch&logoColor=white

Official implementation of "No-rank Tensor Decomposition Using Metric Learning" by Maryam Bagherian. This repository contains the code for tensor decomposition framework that replaces traditional reconstruction objectives with discriminative, similarity-based optimization using metric learning.

# 📝 Abstract
Tensor decomposition faces fundamental challenges in analyzing high-dimensional data, where traditional methods based on reconstruction and fixed-rank constraints often fail to capture semantically meaningful structures. This paper introduces a no-rank tensor decomposition framework grounded in metric learning, which replaces reconstruction objectives with a discriminative, similarity-based optimization.

The proposed approach learns data-driven embeddings by optimizing a triplet loss with diversity and uniformity regularization, creating a feature space where distance directly reflects semantic similarity. We provide theoretical guarantees for the framework's convergence and establish bounds on its metric properties.

# Key contributions:

- Metric learning approach to tensor decomposition

- No-rank formulation that avoids fixed-rank constraints

- Theoretical convergence guarantees

- Superior performance with smaller training datasets

- Efficient alternative for domains with limited labeled data

# 📊 Results
The proposed method demonstrates substantial improvements across diverse domains:

- Domain	Dataset	Improvement over Baselines
- Face Recognition	LFW, Olivetti	+15-25% in clustering metrics
- Brain Connectivity	ABIDE	+20-30% in separation metrics
- Simulated Data	Galaxy morphology, Crystal structures	+25-35% in semantic alignment
  
# Key findings:

- Outperforms: PCA, t-SNE, UMAP, CP, and Tucker decompositions

- Clustering metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Separation Ratio, Adjusted Rand Index, Normalized Mutual Information

- Trade-off revelation: Metric learning optimizes global class separation while deliberately transforming local geometry to align with semantic relationships

- Data efficiency: Achieves superior performance with smaller training datasets compared to transformer-based methods

# 🚀 Quick Start
Installation
```bash
# Clone the repository
git clone https://github.com/mbagherian/No-Rank-Tensor-Decomposition.git
cd No-Rank-Tensor-Decomposition

# Create and activate conda environment (optional)
conda create -n tensor-metric python=3.9
conda activate tensor-metric

# Install dependencies
pip install -r requirements.txt
```

# Basic Usage
```python
import torch
from face_metric_learning import main

# Run the main evaluation (will prompt for dataset selection)
main()
```
# Requirements
```txt
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tensorly>=0.8.0
umap-learn>=0.5.0
pandas>=1.4.0
scipy>=1.8.0
```
# 📁 Project Structure
```text
no-rank-tensor-decomposition/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── main.py                           # Main execution script
├── face_metric_learning.py           # Core implementation
├── examples/
│   ├── lfw_example.py                # LFW dataset example
│   ├── olivetti_example.py           # Olivetti faces example
│   └── brain_connectivity.py         # ABIDE dataset example (placeholder)
└── results/
    ├── figures/                      # Generated visualizations
    └── metrics/                      # Evaluation results
```
# 🔧 Core Components
1. Metric Learning Model
```python
class FaceMetricLearningModel(nn.Module):
    def __init__(self, input_shape, latent_dim=128):
        super().__init__()
        # Convolutional encoder for feature extraction
        self.conv_encoder = nn.Sequential(...)
        
        # Projection head for metric learning
        self.projection_head = nn.Sequential(...)
```
2. Improved Loss Function
```python
class ImprovedFaceMetricLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.3, beta=0.2):
        super().__init__()
        # Triplet loss + local preservation + diversity regularization
        self.triplet_loss = ...
        self.local_preservation_loss = ...
```
3. Tensor Decomposition Comparison
```python
def compare_tensor_decomposition_methods(original_data, labels, metric_embeddings, dataset_name):
    """
    Comprehensive comparison of:
    - Original Data + K-Means
    - PCA + K-Means
    - t-SNE + K-Means
    - UMAP + K-Means
    - CP Decomposition (various ranks)
    - Tucker Decomposition (various ranks)
    - Our Metric Learning approach
    """
```
# 📈 Evaluation Metrics
Our framework evaluates across 7 key metrics:

1- Clustering Quality

- Silhouette Score

- Davies-Bouldin Index

- Adjusted Rand Index (ARI)

- Normalized Mutual Information (NMI)

2- Structural Preservation

- Separation Ratio

- Continuity

- Trustworthiness

3- Reconstruction Quality (for comparative methods)

- Reconstruction Error

- Explained Variance

# 🎯 Usage Examples
Example 1: LFW Faces Dataset
```python
from face_metric_learning import EnhancedFaceDataset, main

# Load LFW dataset
dataset = EnhancedFaceDataset(min_faces_per_person=70, resize=0.4)

# Train metric learning model
model, losses = add_regularization_to_training(dataset, num_epochs=100)

# Evaluate and compare with tensor methods
embeddings, labels = evaluate_face_embeddings(model, dataset)
results = compare_tensor_decomposition_methods(dataset.images, labels, embeddings, "LFW Faces")
```
Example 2: Olivetti Faces Dataset
```python
from face_metric_learning import OlivettiFaceDataset

dataset = OlivettiFaceDataset()
# ... (same training and evaluation pipeline)
```
# 📊 Visualization Outputs
The code generates comprehensive visualizations:

1- Manifold visualizations of embeddings from different methods

2- Reconstruction performance comparison charts

3- Structural metrics bar charts

4- Sample reconstructions for qualitative comparison

5- Training loss curves

# 🔬 Key Features
1- No-rank Formulation
Avoids fixed-rank constraints that limit traditional tensor decomposition

Adapts to inherent data dimensionality

2- Data-driven Embeddings
Learns semantic similarity directly from data

Optimizes for downstream tasks (clustering, classification)

3- Computational Efficiency
Faster convergence than transformer-based methods

Effective with limited labeled data

GPU-accelerated training

4-  Flexible Architecture
Modular design for easy extension to new domains

Compatible with various neural network architectures

# 📝 Citation
If you use this code or paper in your research, please cite:

bibtex
@article{bagherian2025norank,
  title={No-rank Tensor Decomposition Using Metric Learning},
  author={Bagherian, Maryam},
  journal={arXiv preprint [arXiv:2511.01816](https://arxiv.org/abs/2511.01816)},
  year={2025}
}
# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

# 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# 📧 Contact
For questions or discussions about this work, please open an issue on GitHub or contact the author.

🔗 Links
[arXiv Paper] (https://arxiv.org/abs/2511.01816)

[GitHub Repository] (https://github.com/mbagherian/No-Rank-Tensor-Decomposition/)

[TensorLy Documentation] (https://tensorly.org/stable/index.html)


# The project should include the following files:
```text
no-rank-tensor-decomposition/
├── .gitignore          
├── README.md           
├── requirements.txt    
├── setup.py           
├── LICENSE            
├── main.py            
├── metric_learning/
│   ├── __init__.py
│   ├── models.py
│   ├── datasets.py
│   ├── losses.py
│   ├── decomposition.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── utils.py
│   └── main.py
├── examples/
│   ├── run_lfw.py
│   ├── run_olivetti.py
│   └── compare_methods.py
└── scripts/
    └── train.py
```
