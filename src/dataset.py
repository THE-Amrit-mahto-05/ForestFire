import torch
from torch.utils.data import Dataset
import numpy as np

class FireDataset(Dataset):
    def __init__(self, feature_path, label_path, tile_size=128):
        self.features = np.load(feature_path) 
        self.labels = np.load(label_path)     
        self.tile_size = tile_size
        
        self.h_tiles = self.features.shape[1] 
        self.w_tiles = self.features.shape[2] 

    def __len__(self):
        return self.h_tiles * self.w_tiles

    def __getitem__(self, idx):
        i = (idx//self.w_tiles)*self.tile_size
        j = (idx%self.w_tiles)*self.tile_size
        
        tile_x = self.features[:,i:i+self.tile_size,j:j+self.tile_size]
        tile_y = self.labels[i:i+self.tile_size,j:j+self.tile_size]
        return torch.from_numpy(tile_x).float(), torch.from_numpy(tile_y).float().unsqueeze(0)