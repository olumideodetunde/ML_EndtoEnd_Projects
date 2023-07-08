# -*- coding: utf-8 -*-
import os
import glob
import click
import random
import shutil
import logging
import pandas as pd
from pathlib import Path

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path(exists=True))  

class DataCreator:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath =  input_filepath
        self.output_filepath = output_filepath
        self.labels = pd.read_csv(os.path.join(self.input_filepath, 'labels.csv'))

    def create_directories(self):
        self.train_dir = os.path.join(self.output_filepath, "train")
        self.val_dir = os.path.join(self.output_filepath, "val")
        self.test_dir = os.path.join(self.output_filepath,"test")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        return self.train_dir, self.val_dir, self.test_dir
    
    def load_images(self):
        images = glob.glob(self.input_filepath + '/*.jpg')
        return images
    
    def shuffle_images(self,):
        images = self.load_images()
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
    
    def _copy_image(self, image_list, dir):
        updated_image_list = []
        for image_path in image_list:
            destination_path = os.path.join(self.output_filepath, dir, os.path.basename(image_path))
            shutil.copy(image_path, destination_path)
            updated_image_list.append(destination_path)
        return updated_image_list
            
    def copy_images_to_dir(self):
        train_images, val_images, test_images = self.split_images()
        self.train_dest = self._copy_image(train_images, dir="train")
        self.val_dest = self._copy_image(val_images, dir="val")
        self.test_dest = self._copy_image(test_images, dir="test")
        return self.train_dest, self.val_dest, self.test_dest
            
    def _get_labels(self, image_paths):
        image_names = [os.path.basename(image) for image in image_paths]
        labels = self.labels[self.labels['image'].isin(image_names)].reset_index(drop=True)
        return labels
    
    def save_labels(self):
        train_label = self._get_labels(self.train_dest)
        val_label = self._get_labels(self.val_dest)
        test_label = self._get_labels(self.test_dest)
        train_label.to_csv(os.path.join(self.train_dir, 'label.csv'))
        val_label.to_csv(os.path.join(self.val_dir, 'label.csv'))
        test_label.to_csv(os.path.join(self.test_dir, 'label.csv'))
        return train_label, val_label, test_label

    def create_computervision_data(self):
        self.create_directories()
        self.load_images()
        self.split_images()
        self.copy_images_to_dir()
        self.save_labels()

@click.command()
@click.option('--input_filepath', type=click.Path(exists=True), required=True)
@click.option('--output_filepath', type=click.Path(), required=True)     
def main (input_filepath, output_filepath):
    dataset = DataCreator(input_filepath, output_filepath)
    dataset.create_computervision_data()
    logger = logging.getLogger(__name__)
    logger.info('making computervision ready dataset')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()