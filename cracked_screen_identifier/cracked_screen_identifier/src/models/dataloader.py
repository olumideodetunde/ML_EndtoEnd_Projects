# -*- coding: utf-8 -*-
import os
import cv2
import torch
import click
import logging
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PhoneDataset(Dataset):
    
    def __init__(self, data_dir, label_csvfile): 
        self.data_dir = data_dir
        self.label_csvfile = pd.read_csv(label_csvfile)
        self.label_csvfile['label_encoded'] = self.label_csvfile['label'].apply(lambda x: 0 if x == 'perfect' else 1)
        
    def __len__(self):
        return len(self.label_csvfile)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, str(self.label_csvfile.iloc[idx,1]))
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        image =  image.permute(2,1,0)
        label = self.label_csvfile.iloc[idx,3] #label_encoded
        return image,label
    
if __name__ == "__main__":
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('making dataset and dataloader from val and train dir')
    
    train_data_dir = "data/processed/train"
    train_csv = "data/processed/train/label.csv"
    train_dataset = PhoneDataset(train_data_dir, train_csv)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_data_dir = "data/processed/val"
    val_csv = "data/processed/val/label.csv"
    val_dataset = PhoneDataset(val_data_dir, val_csv)    
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    
    
