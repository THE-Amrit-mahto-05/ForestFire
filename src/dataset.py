import torch
from torch.utils.data import Dataset
import numpy as np
import os

class FireDataset(Dataset):
    def __init__(self, feature_path, label_path, tile_size=256, transform=None):
        """
        PyTorch Dataset for Geospatial Fire Risk.
        Loads large numpy stacks and returns tiles.
        """
        self.features = np.load(feature_path) # Shape: (C, H, W)
        self.labels = np.load(label_path)     # Shape: (H, W) or (1, H, W)
        
        if self.labels.ndim == 2:
            self.labels = self.labels[np.newaxis, ...]
            
        self.tile_size = tile_size
        self.transform = transform
        
        self.C, self.H, self.W = self.features.shape
        
        # Calculate number of tiles
        self.n_tiles_h = self.H // tile_size
        self.n_tiles_w = self.W // tile_size
        self.total_tiles = self.n_tiles_h * self.n_tiles_w

    def __len__(self):
        return self.total_tiles

    def __getitem__(self, idx):
        # Determine tile coordinates
        h_idx = idx // self.n_tiles_w
        w_idx = idx % self.n_tiles_w
        
        y1 = h_idx * self.tile_size
        y2 = y1 + self.tile_size
        x1 = w_idx * self.tile_size
        x2 = x1 + self.tile_size
        
        feature_tile = self.features[:, y1:y2, x1:x2]
        label_tile = self.labels[:, y1:y2, x1:x2]
        
        # Convert to tensor
        feature_tensor = torch.from_numpy(feature_tile).float()
        label_tensor = torch.from_numpy(label_tile).float()
        
        if self.transform:
            feature_tensor = self.transform(feature_tensor)
            
        return feature_tensor, label_tensor

def get_dataloader(feature_path, label_path, batch_size=4, tile_size=256, shuffle=True):
    dataset = FireDataset(feature_path, label_path, tile_size=tile_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)