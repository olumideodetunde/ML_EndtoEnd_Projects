# -*- coding: utf-8 -*-
import os
import cv2
import glob
import click
import logging
import pandas as pd
from model import ModelNet
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
#@click.argument("output_filepath", type=click.Path())

## ---- Class Dataset -----
class ScreenDataset(Dataset):
    
    def __init__(self, data_dir, label_csv, transform):
        
        self.data_dir = data_dir
        self.label_csv = pd.read_csv(label_csv)
        self.label_csv
        self.label_csv['class_encoded'] = self.label_csv.apply(lambda x: 0 if x['label'] == "perfect" else 1, axis=1)
        self.transform = transform
    
    def __len__(self):
        return len(self.label_csv)
    
    def __getitem__(self, index):
        
        filename = os.path.join(self.data_dir, self.label_csv.iloc[index, 1])
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = self.label_csv.iloc[index, 3]
        return image, label
    

def main(input_filepath):
    
    #train dataset
    train_dir = os.path.join(input_filepath, "train")
    label_csv = os.path.join(train_dir, "label.csv")
    train_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),])
    train_dataset = ScreenDataset(train_dir, label_csv, train_transform)
    
    #validation dataset
    val_dir = os.path.join(input_filepath, "val")
    label_csv = os.path.join(val_dir, "label.csv")
    val_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),])
    val_dataset = ScreenDataset(val_dir, label_csv, val_transform)
    
    print("completely executed")
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
    