# -*- coding: utf-8 -*-
import os
import glob
import click
import random
import shutil
import logging
import pandas as pd
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

class DataCreator:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath =  input_filepath
        self.output_filepath = output_filepath
        self.labels = pd.read_csv(os.path.join(self.input_filepath, 'labels.csv'))

    def create_directories(self):
        train_dir = os.path.join(self.output_filepath, "train")
        val_dir = os.path.join(self.output_filepath, "val")
        test_dir = os.path.join(self.output_filepath,"test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        return train_dir, val_dir, test_dir
    
    def load_images(self):
        images = glob.glob(self.input_filepath + '/*.jpg')
        return images
    
    def shuffle_images(self,):
        images = self.load_data()
        for i in range (4):    #small number of shuffles due to large files
            random.shuffle(images)
        return images
    
    def split_images(self):
        images = self.shuffle_images()
        split_1 = int(0.8 * len(images))
        split_2 = int(0.95 * len(images))
        train_images = images[:split_1]
        val_images  = images [split_1:split_2]
        test_images =  images [split_2:]
        return train_images, val_images, test_images
    
    def _get_labels(self, image_paths):
        image_names = [os.path.basename(image) for image in image_paths]
        labels = self.labels[self.labels['image'].isin(image_names)]
        return labels

    def curate_labels(self):
        train_images, val_images, test_images = self.split_images()
        train_label = self._get_labels(train_images)
        val_label = self._get_labels(val_images)
        test_label = self._get_labels(test_images)
        return train_label, val_label, test_label
            
    def _get_final_image_path(self, image_list, directory):
        for i, image in enumerate(image_list):
            image_basename = os.path.basename(image)
            final_image_path = os.path.join(directory, image_basename)
            image_list[i] = final_image_path
        return image_list

    def copy_images_to_dirs(self):
        train_dir, val_dir, test_dir = self.create_directories()
        train_images, val_images, test_images = self.split_images()
        train_path = self._get_final_image_path(train_images, train_dir)
        val_path = self._get_final_image_path(val_images, val_dir)
        test_path = self._get_final_image_path(test_images, test_dir)
        shutil.copy(original_image_path, train_path)
        
    def save_labels_to_dir(self):
        train_label, val_label, test_label = self.curate_labels()
        train_dir, val_dir, test_dir =  self.create_directories
        train_label.to_csv(os.path.join(train_dir, 'label.csv'))
        val_label.to_csv(os.path.join(val_dir, 'label.csv'))
        test_label.to_csv(os.path.join(test_dir, 'label.csv'))
        
    def create_computervision_data(self):
        self.load_images()
        self.shuffle_images()
        self.split_images()
        self.copy_images_to_dirs()
        self.curate_labels()
        self.save_labels_to_dir()
         
def main (input_filepath, output_filepath):
    dataset = DataCreator(input_filepath, output_filepath)
    dataset.create_computervision_data()
    
if __name__ == "__main__":
    main()