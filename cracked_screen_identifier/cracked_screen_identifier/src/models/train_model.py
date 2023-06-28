# -*- coding: utf-8 -*-
import os
import cv2
import copy
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from model import ModelNet
from torch.utils.tensorboard import SummaryWriter

def main():
##--------- Prepare the data-------------
    epochs = 20
    batch_size = 16
    class PhoneDataset(Dataset):
        def __init__(self, dir, csv_file, transform): 
            
            self.dir = dir
            self.csv_file = pd.read_csv(csv_file)
            self.csv_file['label_encoded'] = self.csv_file['label'].apply(lambda x: 0 if x == 'perfect' else 1)
            self.transform = transform
            
        def __len__(self):
            return len(self.csv_file)
        
        def __getitem__(self, idx):
            image_name = os.path.join(self.dir, str(self.csv_file.iloc[idx,1]))
            image = cv2.imread(image_name)
            image =  torch.from_numpy(image).permute(2,0,1)
            label = self.csv_file.iloc[idx,3] #label_encoded
            return image,label

    # Create an instances of the dataset class
    train_dir = "data/processed/train"
    train_csv = "data/processed/train/label.csv"
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = PhoneDataset(train_dir, train_csv, transform=train_transform)

    val_dir = "data/processed/val"
    val_csv = "data/processed/val/label.csv"
    val_transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = PhoneDataset(val_dir, val_csv, transform=val_transform)

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),])
    val_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),])

    #Update the transforms
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#--------Trainig loop and exp tracking ---------------
    run_name = f"Run 1 -  Baseline Model"
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + run_name
    tb = SummaryWriter(log_dir=log_dir) #Create tensorboard object

    model = ModelNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(1, epochs+1):
        
        train_loss = 0.0
        valid_loss = 0.0
        
        model.train()
        for data, label in train_loader:
            data = data.float()
            #zero the gradients
            optimizer.zero_grad()
            #forward pass
            output = model(data.float())
            #Calculate batch loss
            loss = criterion(output, label)
            #backward pass
            loss.backward()
            #Optimize the weights
            optimizer.step()
            #update training loss
            train_loss += loss.item() * data.size(0)
            
        #validate the model
        model.eval()
        for data, label in val_loader:
            data = data.float()
            output = model(data)
            loss = criterion(output, label)
            valid_loss += loss.item() * data.size(0)
            
        #Calculate the average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(val_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        tb.add_scalars("Loss", {"Train": train_loss, "Val": valid_loss}, epoch)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        #Save the model if validation loss has decreased
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'{log_dir}/modelnet.pth')
            print("Saved the new best model")
    tb.close()
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()