# no-rank-tensor-decomposition/metric_learning/utils.py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

from .models import FaceMetricLearningModel, device
from .losses import ImprovedFaceMetricLoss
from .datasets import EnhancedFaceDataset, OlivettiFaceDataset

def improved_smart_triplet_mining(embeddings, labels, original_data=None, k=5):
    """Triplet mining that considers local structure preservation"""
    anchors, positives, negatives = [], [], []
    
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()
    
    # If original data is provided, use it to find local neighbors
    if original_data is not None:
        original_flat = original_data.reshape(original_data.shape[0], -1)
        nbrs_original = NearestNeighbors(n_neighbors=k+1).fit(original_flat)
        _, original_neighbors = nbrs_original.kneighbors(original_flat)
    
    nbrs_embedding = NearestNeighbors(n_neighbors=len(embeddings_np)).fit(embeddings_np)
    _, embedding_neighbors = nbrs_embedding.kneighbors(embeddings_np)
    
    for i in range(len(embeddings)):
        anchor = embeddings[i]
        anchor_label = labels_np[i]
        
        # Find positive samples that are local neighbors in original space
        if original_data is not None:
            # Use original space neighbors as positive candidates
            local_positives = [j for j in original_neighbors[i][1:] 
                             if labels_np[j] == anchor_label and j != i]
        else:
            local_positives = [j for j in range(len(embeddings)) 
                             if labels_np[j] == anchor_label and j != i]
        
        if not local_positives:
            continue
            
        # Choose positive that maintains local structure
        positive_idx = local_positives[0]  # simplest: first local positive
        positive = embeddings[positive_idx]
        
        # Find negative samples that break local structure the most
        # These are points that are close in embedding but should be far
        neg_candidates = [j for j in range(len(embeddings)) 
                        if labels_np[j] != anchor_label]
        
        if not neg_candidates:
            continue
            
        # Find negatives that are neighbors in embedding space but not same class
        embedding_neighbor_negatives = [j for j in embedding_neighbors[i][1:k+1] 
                                      if labels_np[j] != anchor_label]
        
        if embedding_neighbor_negatives:
            # These are the worst offenders for trustworthiness
            negative_idx = embedding_neighbor_negatives[0]
        else:
            # Fallback: random negative
            negative_idx = np.random.choice(neg_candidates)
        
        negative = embeddings[negative_idx]
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    
    if len(anchors) > 0:
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    else:
        return get_simple_triplets(embeddings, labels)

def get_simple_triplets(embeddings, labels):
    """Simple but stable triplet mining"""
    anchors, positives, negatives = [], [], []
   
    for i in range(len(embeddings)):
        anchor = embeddings[i]
        anchor_label = labels[i]
       
        # Find any positive (different from anchor)
        pos_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
        pos_indices = torch.where(pos_mask)[0]
        if len(pos_indices) > 0:
            pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
            positive = embeddings[pos_idx]
        else:
            continue  # Skip if no positive found
           
        # Find any negative
        neg_mask = labels != anchor_label
        neg_indices = torch.where(neg_mask)[0]
        if len(neg_indices) > 0:
            neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
            negative = embeddings[neg_idx]
        else:
            continue  # Skip if no negative found
           
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
   
    if len(anchors) == 0:
        # Emergency fallback - use random pairs
        idx1, idx2, idx3 = torch.randint(0, len(embeddings), (3,))
        return embeddings[idx1:idx1+1], embeddings[idx2:idx2+1], embeddings[idx3:idx3+1]
   
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def evaluate_face_embeddings(model, dataset):
    """Evaluate the model and return embeddings"""
    model.eval()
    embeddings = []
    all_labels = []
   
    with torch.no_grad():
        for data, label in dataset:
            data = data.unsqueeze(0).to(device)
            emb, _ = model(data)
            embeddings.append(emb.cpu())
            all_labels.append(label)
   
    embeddings = torch.cat(embeddings, dim=0)
    labels = np.array(all_labels)
   
    return embeddings, labels

def add_regularization_to_training(dataset, num_epochs=100, batch_size=16):
    """Train the metric learning model with regularization"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    sample, _ = dataset[0]
    input_shape = sample.shape[1:]
    
    model = FaceMetricLearningModel(input_shape=input_shape, latent_dim=128).to(device)
    
    # Use improved loss
    criterion = ImprovedFaceMetricLoss(margin=1.0, alpha=0.3)
    
    # More conservative optimizer
    optimizer = optim.AdamW([
        {'params': model.conv_encoder.parameters(), 'lr': 1e-5},  # Lower LR for encoder
        {'params': model.projection_head.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    
    # Prepare original data for smart mining
    original_data = np.array([dataset.images[i] for i in range(len(dataset))])
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch, batch_labels in dataloader:
            batch = batch.to(device)
            batch_labels = batch_labels.to(device)
            
            embeddings, _ = model(batch)
            
            # Get batch-specific original data
            batch_original = original_data[[i % len(dataset) for i in range(len(batch))]]
            
            # Use improved triplet mining
            anchor_emb, positive_emb, negative_emb = improved_smart_triplet_mining(
                embeddings, batch_labels, batch_original)
            
            optimizer.zero_grad()
            loss = criterion(anchor_emb, positive_emb, negative_emb, embeddings)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            epoch_loss = total_loss / num_batches
            train_losses.append(epoch_loss)
            scheduler.step()
            
            if epoch % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Loss {epoch_loss:.4f}, LR: {current_lr:.2e}")
    
    return model, train_losses