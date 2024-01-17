# -*- coding: utf-8 -*-
import os
import cv2
import json
import copy
import click
import torch
import logging
from datetime import datetime
from dataloader import PhoneDataset
from model_architecture import ModelNet
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = ModelNet()
        self.criterion =  torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.tb =  None
        self.experiment_name = f"Experiment 2 - Baseline Model"
        self.log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + self.experiment_name
        
    def prepare_dataset(self,train_data_dir, val_data_dir):
        train_csv = train_data_dir + "/label.csv"
        train_dataset = PhoneDataset(train_data_dir, train_csv)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_csv = val_data_dir + "/label.csv"
        val_dataset = PhoneDataset(val_data_dir, val_csv)    
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)
        return train_loader, val_loader
    
    def train_step(self, image, label):
        image = image.float()
        self.optimizer.zero_grad()
        output = self.model(image)
        loss =  self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item() * image.size(0)
    
    def train_model(self):
        self.tb = SummaryWriter(log_dir=self.log_dir)
        train_loader, val_loader = self.prepare_dataset(train_data_dir="data/processed/train", 
                                                        val_data_dir="data/processed/val")
        
        for epoch in range(1, self.epochs+1):
            train_loss = 0.0
            val_loss = 0.0
            self.model.train()
            for image, label in train_loader:
                train_loss += self.train_step(image, label)
            self.model.eval()
            for image, label in val_loader:
                val_loss +=  self.train_step(image, label)
            train_loss =  train_loss / len(train_loader.sampler)
            val_loss =  val_loss / len(val_loader.sampler)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.tb.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model, f'{self.log_dir}/modelnet.pth')
                print("Saved the new best model")
        self.tb.close()
        
    def save_training_parameters(self):
        parameters = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "model": self.model.__class__.__name__,
        }
        with open(f'{self.log_dir}/parameters.json', 'w') as fp:
            json.dump(parameters, fp)
            
    def run_training(self):
        self.train_model()
        self.save_training_parameters()

@click.command()
@click.option("--epochs", default=10, help="Number of epochs")
@click.option("--learning-rate", default=0.001, help="Learning rate")
def main(epochs,  learning_rate):
    trainer = Trainer(epochs, learning_rate)
    trainer.run_training()

if __name__ == "__main__":
    main()
