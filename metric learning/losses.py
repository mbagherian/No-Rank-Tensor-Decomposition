# no-rank-tensor-decomposition/metric_learning/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedFaceMetricLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.3, beta=0.2):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for local preservation
        self.beta = beta    # Weight for diversity
        
    def triplet_loss(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()
    
    def local_preservation_loss(self, embeddings, k=5):
        """Encourage local neighborhood preservation"""
        pairwise_dists = torch.cdist(embeddings, embeddings, p=2)
        
        # Get k-nearest neighbors
        _, indices = torch.topk(pairwise_dists, k=k+1, largest=False, dim=1)
        
        loss = 0.0
        for i in range(len(embeddings)):
            # Compare distances to neighbors vs non-neighbors
            neighbors = indices[i, 1:]  # exclude self
            non_neighbors = torch.tensor([j for j in range(len(embeddings)) 
                                        if j != i and j not in neighbors]).to(embeddings.device)
            
            if len(non_neighbors) > 0:
                neighbor_dists = pairwise_dists[i, neighbors]
                non_neighbor_dists = pairwise_dists[i, non_neighbors[:k]]  # sample k non-neighbors
                
                # Encourage neighbor distances to be smaller than non-neighbor distances
                loss += torch.clamp(neighbor_dists.mean() - non_neighbor_dists.mean() + 0.1, min=0.0)
        
        return loss / len(embeddings)
    
    def forward(self, anchor, positive, negative, embeddings=None):
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        
        if embeddings is not None:
            local_loss = self.local_preservation_loss(embeddings)
            return triplet_loss + self.alpha * local_loss
        
        return triplet_loss