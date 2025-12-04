# no-rank-tensor-decomposition/scripts/train.py
#!/usr/bin/env python3
"""
Command-line script for training the metric learning model.
Usage: python scripts/train.py --dataset lfw --epochs 100
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric_learning import (
    EnhancedFaceDataset,
    OlivettiFaceDataset,
    add_regularization_to_training,
    evaluate_face_embeddings
)

def main():
    parser = argparse.ArgumentParser(description='Train metric learning model')
    parser.add_argument('--dataset', type=str, default='lfw',
                       choices=['lfw', 'olivetti'],
                       help='Dataset to use (lfw or olivetti)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--output', type=str, default='model.pth',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == 'lfw':
        print(f"Loading LFW dataset...")
        dataset = EnhancedFaceDataset(min_faces_per_person=70, resize=0.4)
        dataset_name = "LFW Faces"
    else:
        print(f"Loading Olivetti dataset...")
        dataset = OlivettiFaceDataset()
        dataset_name = "Olivetti Faces"
    
    # Train model
    print(f"Training {dataset_name} model for {args.epochs} epochs...")
    model, train_losses = add_regularization_to_training(
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate
    embeddings, labels = evaluate_face_embeddings(model, dataset)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_shape': dataset[0][0].shape[1:],
        'latent_dim': 128,
        'dataset': args.dataset,
        'embeddings': embeddings,
        'labels': labels
    }, args.output)
    
    print(f"✅ Model saved to {args.output}")

if __name__ == "__main__":
    main()