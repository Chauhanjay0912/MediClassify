import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label_encoder=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        if label_encoder is None:
            self.le = LabelEncoder()
            self.labels = self.le.fit_transform(self.df['diagnostic'])
        else:
            self.le = label_encoder
            self.labels = self.le.transform(self.df['diagnostic'])
        
        self.classes = self.le.classes_
        self.valid_indices = self._validate_images()
    
    def _validate_images(self):
        valid_indices = []
        for idx in range(len(self.df)):
            img_name = self.df.iloc[idx]['img_id']
            img_path = os.path.join(self.img_dir, f"{img_name}.png")
            if os.path.exists(img_path):
                valid_indices.append(idx)
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_name = self.df.iloc[actual_idx]['img_id']
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        
        image = Image.open(img_path).convert('RGB')
        label = self.labels[actual_idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
