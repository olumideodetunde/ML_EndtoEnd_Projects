# -*- coding: utf-8 -*-
import os
import cv2
import torch
import pandas as pd
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