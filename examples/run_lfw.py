# no-rank-tensor-decomposition/examples/run_lfw.py
#!/usr/bin/env python3
"""
Example script to run the LFW faces dataset experiment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric_learning import (
    EnhancedFaceDataset,
    add_regularization_to_training,
    evaluate_face_embeddings,
    compare_tensor_decomposition_methods
)
import matplotlib.pyplot as plt

def main():
    """Run LFW faces experiment"""
    print("🚀 Running LFW Faces Dataset Experiment")
    print("="*60)
    
    # Load dataset
    dataset = EnhancedFaceDataset(min_faces_per_person=70, resize=0.4)
    
    # Train model
    print("\n🎯 Training Metric Learning Model...")
    model, train_losses = add_regularization_to_training(
        dataset, 
        num_epochs=100,
        batch_size=16
    )
    
    # Evaluate
    print("\n📊 Evaluating Embeddings...")
    embeddings, labels = evaluate_face_embeddings(model, dataset)
    
    # Compare methods
    print("\n🔬 Comparing with Tensor Decomposition Methods...")
    results_df, methods_dict = compare_tensor_decomposition_methods(
        dataset.images, 
        labels, 
        embeddings, 
        "LFW Faces"
    )
    
    # Display best results
    best_method = results_df.loc[results_df['Silhouette'].idxmax()]
    print(f"\n🏆 BEST METHOD: {best_method['Method']}")
    print(f"   Silhouette Score: {best_method['Silhouette']:.4f}")
    print(f"   ARI: {best_method['ARI']:.4f}")
    print(f"   NMI: {best_method['NMI']:.4f}")
    
    # Plot training loss
    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Metric Learning Training Loss (LFW Faces)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/lfw_training_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()