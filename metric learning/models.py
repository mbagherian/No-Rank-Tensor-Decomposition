# no-rank-tensor-decomposition/metric_learning/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceMetricLearningModel(nn.Module):
    def __init__(self, input_shape, latent_dim=128):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
       
        # Convolutional encoder for faces
        self.conv_encoder = nn.Sequential(
            # Input: (1, H, W)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
           
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
           
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
       
        # Calculate conv output size
        with torch.no_grad():
            sample = torch.zeros(1, 1, *input_shape)
            conv_out = self.conv_encoder(sample)
            conv_flatten_size = conv_out.view(1, -1).shape[1]
       
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(conv_flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
       
    def forward(self, x):
        # Convolutional features
        features = self.conv_encoder(x)
        features = features.view(features.size(0), -1)
       
        # Project to embedding space
        z = self.projection_head(features)
        return F.normalize(z, p=2, dim=1), features