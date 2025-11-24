import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, validate_files=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        required = {'name', 'age', 'gender'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_file}: {missing}")
        if validate_files:
            self._validate_images()

    def _validate_images(self):
        valid_mask = self.df['name'].apply(
            lambda x: os.path.exists(os.path.join(self.img_dir, x))
        )
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {
            'age':    torch.tensor(int(row['age']), dtype=torch.float32),
            'gender': torch.tensor(int(row['gender']), dtype=torch.long)
        }
        return image, sample
    
    def get_class_distribution(self, column='gender'):
        return self.df[column].value_counts().to_dict()
    
    def get_stats(self):
        stats = {
            'total_samples': len(self.df),
            'age_mean': self.df['age'].mean(),
            'age_std': self.df['age'].std(),
            'age_min': self.df['age'].min(),
            'age_max': self.df['age'].max(),
            'gender_distribution': self.df['gender'].value_counts().to_dict()
        }
        return stats