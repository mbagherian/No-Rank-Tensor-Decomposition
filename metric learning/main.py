# no-rank-tensor-decomposition/metric_learning/main.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import warnings
warnings.filterwarnings('ignore')

from .decomposition import (
    ReconstructionAnalyzer, 
    cp_decomposition_embeddings, 
    tucker_decomposition_embeddings
)
from .utils import add_regularization_to_training, evaluate_face_embeddings
from .datasets import EnhancedFaceDataset, OlivettiFaceDataset
from .metrics import calculate_all_metrics
from .visualization import (
    visualize_tensor_comparison_with_reconstruction,
    visualize_tensor_decomposition_comparison
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compare_tensor_decomposition_methods(original_data, labels, metric_embeddings, dataset_name):
    """Compare tensor decomposition methods with metric learning"""
    print(f"\n🔬 COMPARING TENSOR DECOMPOSITION METHODS FOR {dataset_name}")
    print("="*70)
    
    original_flat = original_data.reshape(original_data.shape[0], -1)
    metric_emb = metric_embeddings.cpu().numpy() if isinstance(metric_embeddings, torch.Tensor) else metric_embeddings
    
    # Get tensor decomposition results
    cp_results = cp_decomposition_embeddings(original_data)
    tucker_results = tucker_decomposition_embeddings(original_data)
    
    # Calculate reconstruction metrics for metric learning
    reconstructor = ReconstructionAnalyzer()
    ml_reconstructed = reconstructor.reconstruct_from_metric_learning(metric_embeddings, original_data)
    ml_metrics = reconstructor.calculate_reconstruction_metrics(original_data, ml_reconstructed)
    
    methods = {
        'Original Data': {'embeddings': original_flat},
        'PCA': {'embeddings': PCA(n_components=20).fit_transform(original_flat)},
        't-SNE': {'embeddings': TSNE(n_components=2, random_state=42).fit_transform(original_flat)},
        'UMAP': {'embeddings': umap.UMAP(random_state=42).fit_transform(original_flat)},
        'Metric Learning': {
            'embeddings': metric_emb,
            'reconstruction_metrics': ml_metrics,
            'reconstructed': ml_reconstructed
        }
    }
    
    # Add CP decomposition results
    for method_name, result in cp_results.items():
        if result is not None:
            methods[method_name] = result
    
    # Add Tucker decomposition results - ensure they're included
    tucker_added = False
    for method_name, result in tucker_results.items():
        if result is not None:
            methods[method_name] = result
            tucker_added = True
    
    if not tucker_added:
        print("⚠️ Warning: No Tucker decomposition results were added to methods")
    
    results = []
    n_clusters = len(np.unique(labels))
    
    for method_name, method_data in methods.items():
        # Skip if data is None (failed decomposition)
        if method_data is None:
            continue
            
        embeddings = method_data['embeddings']
        
        # Calculate all metrics including ARI and NMI
        metrics = calculate_all_metrics(embeddings, labels, original_data, n_clusters)
        
        result_dict = {
            'Method': method_name,
            'Silhouette': metrics['Silhouette'],
            'Davies-Bouldin': metrics['Davies-Bouldin'],
            'Separation_Ratio': metrics['Separation_Ratio'],
            'Continuity': metrics['Continuity'],
            'Trustworthiness': metrics['Trustworthiness'],
            'ARI': metrics['ARI'],
            'NMI': metrics['NMI']
        }
        
        # Add reconstruction metrics if available
        if 'reconstruction_metrics' in method_data:
            result_dict.update({
                'Reconstruction_Error': method_data['reconstruction_metrics']['reconstruction_error'],
                'Explained_Variance': method_data['reconstruction_metrics']['explained_variance']
            })
        else:
            result_dict.update({
                'Reconstruction_Error': None,
                'Explained_Variance': None
            })
        
        results.append(result_dict)
        
        print(f"{method_name:25} Silhouette: {metrics['Silhouette']:.4f}, "
              f"ARI: {metrics['ARI']:.4f}, NMI: {metrics['NMI']:.4f}")
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\n📊 COMPREHENSIVE TENSOR DECOMPOSITION PERFORMANCE:")
    print(results_df.round(4).to_string(index=False))
    
    # Enhanced Visualization
    visualize_tensor_comparison_with_reconstruction(methods, labels, results, dataset_name, original_data)
    
    return results_df, methods

def main():
    """Main function to run the entire pipeline"""
    # Ask the user to select the dataset
    print("🔄 Please choose a dataset:")
    print("1. LFW Dataset (Labeled Faces in the Wild)")
    print("2. Olivetti Faces Dataset")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("🚀 Loading LFW (Labeled Faces in the Wild) Dataset...")
        dataset = EnhancedFaceDataset(min_faces_per_person=70, resize=0.4)
        dataset_name = "LFW Faces"
    elif choice == "2":
        print("🚀 Loading Olivetti Faces Dataset...")
        dataset = OlivettiFaceDataset()
        dataset_name = "Olivetti Faces"
    else:
        print("❌ Invalid input. Please enter either '1' or '2'.")
        return
    
    print("\n🎯 Training Face-Optimized Metric Learning Model...")
    model, train_losses = add_regularization_to_training(dataset, num_epochs=100, batch_size=16)
    
    print("\n📊 Evaluating Face Embeddings...")
    embeddings, labels = evaluate_face_embeddings(model, dataset)
    
    print(f"✅ Metric Learning embeddings shape: {embeddings.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    
    # Compare with tensor decomposition methods
    results_df, methods_dict = compare_tensor_decomposition_methods(
        dataset.images, labels, embeddings, dataset_name
    )
    
    # Debug: Check what methods are available
    print(f"\n🔍 DEBUG: Available methods in methods_dict:")
    for method_name, method_data in methods_dict.items():
        if method_data is not None:
            emb_shape = method_data['embeddings'].shape if 'embeddings' in method_data else 'No embeddings'
            print(f"   {method_name}: {emb_shape}")
        else:
            print(f"   {method_name}: None")
    
    # Optional: Also show the SIMPLE bar chart comparison
    visualize_tensor_decomposition_comparison(results_df, dataset_name)
    
    # Final results summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    best_method = results_df.loc[results_df['Silhouette'].idxmax()]
    print(f"🏆 BEST OVERALL METHOD: {best_method['Method']}")
    print(f"   Silhouette Score: {best_method['Silhouette']:.4f}")
    
    # Show rank sensitivity analysis
    print(f"\n🔍 RANK SENSITIVITY ANALYSIS:")
    cp_methods = [m for m in results_df['Method'] if 'CP' in m]
    tucker_methods = [m for m in results_df['Method'] if 'Tucker' in m]
    
    if cp_methods:
        print("CP Decomposition:")
        for method in cp_methods:
            result = results_df[results_df['Method'] == method].iloc[0]
            print(f"   {method}: Silhouette = {result['Silhouette']:.4f}")
    
    if tucker_methods:
        print("Tucker Decomposition:")
        for method in tucker_methods:
            result = results_df[results_df['Method'] == method].iloc[0]
            print(f"   {method}: Silhouette = {result['Silhouette']:.4f}")
    
    # Plot training loss
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()