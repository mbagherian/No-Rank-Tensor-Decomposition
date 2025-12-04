# no-rank-tensor-decomposition/metric_learning/datasets.py
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces

class EnhancedFaceDataset(Dataset):
    def __init__(self, min_faces_per_person=70, resize=0.4):
        # Use more faces per person for better learning
        self.lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            resize=resize,
            funneled=True,
            slice_=(slice(70, 195), slice(78, 172))  # Tight face crop
        )
       
        self.images = self.lfw_people.images
        self.target = self.lfw_people.target
        self.target_names = self.lfw_people.target_names
       
        # Enhanced preprocessing
        self.images = self.images / 255.0
        self.images = self.standardize_faces(self.images)
       
        print(f"Loaded {len(self.images)} face images from {len(self.target_names)} people")
        print(f"Image shape: {self.images[0].shape}")
        print(f"Samples per person: {np.bincount(self.target)}")
       
    def standardize_faces(self, images):
        """Face-specific standardization"""
        # Per-image standardization (like face recognition models)
        standardized = np.zeros_like(images)
        for i, img in enumerate(images):
            standardized[i] = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return standardized
       
    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.target[idx]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        return image_tensor, label

class OlivettiFaceDataset(Dataset):
    def __init__(self):
        # Load Olivetti Faces dataset
        self.olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
        
        self.images = self.olivetti.images
        self.target = self.olivetti.target
        self.target_names = np.array([f"Person_{i}" for i in range(40)])
        
        # Enhanced preprocessing
        self.images = self.images / 255.0
        self.images = self.standardize_faces(self.images)
        
        print(f"Loaded {len(self.images)} face images from {len(np.unique(self.target))} people")
        print(f"Image shape: {self.images[0].shape}")
        print(f"Samples per person: {np.bincount(self.target)}")
        
    def standardize_faces(self, images):
        """Face-specific standardization"""
        standardized = np.zeros_like(images)
        for i, img in enumerate(images):
            standardized[i] = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return standardized
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.target[idx]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        return image_tensor, label