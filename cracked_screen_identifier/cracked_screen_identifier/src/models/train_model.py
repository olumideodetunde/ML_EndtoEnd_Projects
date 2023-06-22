# -*- coding: utf-8 -*-
#%%
import os
import cv2
import json
import glob
import copy
import torch
import click
import logging
import pandas as pd
from datetime import datetime
from model import ModelNet
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())

## ---- Class Dataset -----
class ScreenDataset(Dataset):
    
    def __init__(self, data_dir, label_csv, transform):
        
        self.data_dir = data_dir
        self.label_csv = pd.read_csv(label_csv)
        # self.label_csv
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
    print(train_dir)
    train_label_csv = os.path.join(train_dir, "label.csv")
    print(train_label_csv)
    train_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    train_dataset = ScreenDataset(train_dir, train_label_csv, train_transform)
    
    #validation dataset
    val_dir = os.path.join(input_filepath, "val")
    val_label_csv = os.path.join(val_dir, "label.csv")
    val_transform = repr(transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]))
    val_dataset = ScreenDataset(val_dir, val_label_csv, val_transform)
    
    #Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader =  DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    #Instantiate the model, criterion, learningrate and optimizer
    model = ModelNet()
    criterion = nn.BCELoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #Setup tensorboard to track the training and val metrics
    experiment_name = F'Exp 1 - Train Model'
    log_dir = "logs" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + experiment_name
    tb = SummaryWriter(log_dir=log_dir)
    
    #Training loop
    epochs = 15
    best_loss = float("inf")
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        val_loss = 0.0
    
        for data, label in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        model.eval()
        for data, label in val_loader:
            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)
        
        #Calculate the average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(val_loader.sampler)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        #Log the loss on tensorboard and print updates
        tb.add_scalars("Loss", {"Train": train_loss, "Val":val_loss}, epoch)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:6f}'.format(epoch, train_loss, val_loss))
        
        #Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'{log_dir}/modelnet.pth')
            #print("Saved the new best model")
    tb.close()
    
    #Save parameters and hyperparameters per training 
    parameters = {
    "epochs": epochs,
    "learning_rate": learning_rate,
    "optimizer": optimizer.__class__.__name__,
    "criterion": criterion.__class__.__name__,
    "model": model.__class__.__name__,}
    with open(f'{log_dir}/parameters.json', 'w') as fp:
        json.dump(parameters, fp)
        
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("data/processed")
