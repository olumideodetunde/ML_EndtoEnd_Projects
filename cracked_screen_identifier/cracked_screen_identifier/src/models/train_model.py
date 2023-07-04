# -*- coding: utf-8 -*-
#%%
import os
import cv2
import json
import copy
import click
import torch
import logging
from datetime import datetime
from model_architecture import ModelNet
from dataloader import main as dataloader_main
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    def __init__(self, epochs, learning_rate):
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = ModelNet()
        self.criterion =  torch.nn.LLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.tb =  None
        
    def train_step(self, image, label):
        image = image.float()
        self.optimizer.zero_grad()
        output = self.model(image)
        loss =  self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item() * image.size(0)
    
    def train_model(self):
        experiment_name = f"Experiment 1 - Baseline Model"
        log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + experiment_name
        self.tb = SummaryWriter(log_dir=log_dir)
        
        for epoch in range(1, self.epochs+1):
            
            train_loss = 0.0
            val_loss = 0.0
            
            self.model.train()
            for image, label in train_loader:
                train_loss = self.train_step(image, label)
            
            self.model.eval()
            for image, label in val_loader:
                val_loss =  self.train_step(image, label)
                
            train_loss =  train_loss / len(train_loader.sampler)
            val_loss =  val_loss / len(val_loader.sampler)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.tb.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model, f'{log_dir}/modelnet.pth')
                print("Saved the new best model")
                
        self.tb.close()
        
        parameters = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "model": self.model.__class__.__name__,
        }

        with open(f'{log_dir}/parameters.json', 'w') as fp:
            json.dump(parameters, fp)
            
    def run_training(self):
        self.train_model()

@click.command()
@click.option("--epochs", default=10, help="Number of epochs")
@click.option("--learning-rate", default=0.001, help="Learning rate")
def main(epochs, batch_size, learning_rate):
    train_loader, val_loader = main("data/processed")
    trainer = Trainer(epochs, batch_size, learning_rate)
    trainer.run_training()

if __name__ == "__main__":
    main()