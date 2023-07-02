# -*- coding: utf-8 -*-
#%%
import os
import cv2
import click
import logging
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

class PhoneDataset(Dataset):
    def __init__(self, data_dir, label_csvfile, transforms=None): 
        
        self.data_dir = data_dir
        self.label_csvfile = pd.read_csv(label_csvfile)
        self.label_csvfile['label_encoded'] = self.label_csvfile['label'].apply(lambda x: 0 if x == 'perfect' else 1)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.label_csvfile)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, str(self.label_csvfile.iloc[idx,1]))
        image = cv2.imread(image_name)
        if self.transforms:
            image =  self.transforms(image)
        image =  image.permute(2,1,0)
        label = self.label_csvfile.iloc[idx,3] #label_encoded
        return image,label

def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
    ])
    return str(transform)

# def main(input_filepath):
    
#     data_dir = input_filepath
#     train_data_dir =  os.path.join(data_dir, "train")
#     train_labelcsv = os.path.join(data_dir, "train/label.csv")
#     train_transform = get_transform()
#     train_dataset = PhoneDataset(train_data_dir, train_labelcsv, transforms=train_transform)
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
#     val_data_dir =  os.path.join(data_dir, "val")
#     val_labelcsv = os.path.join(data_dir, "val/label.csv")
#     val_transform = get_transform()
#     val_dataset = PhoneDataset(val_data_dir, val_labelcsv, transforms=val_transform)    
#     val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    
#     logger = logging.getLogger(__name__)
#     logger.info('making dataset and dataloader from val and train dir')


if __name__ == "__main__":
    
    #main(input_filepath="data/processed/")
    
    train_data_dir = "data/processed/train"
    train_csv = "data/processed/train/label.csv"
    train_dataset = PhoneDataset(train_data_dir, train_csv, get_transform())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_data_dir = "data/processed/val"
    val_csv = "data/processed/val/label.csv"
    val_dataset = PhoneDataset(val_data_dir, val_csv, get_transform())    
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    
    logger = logging.getLogger(__name__)
    logger.info('making dataset and dataloader from val and train dir')
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
# %%
