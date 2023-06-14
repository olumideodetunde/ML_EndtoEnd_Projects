# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
import glob
import cv2
import pandas as pd
import random
import shutil


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    
    ''' Runs data processing scripts to turn raw data from (../interim) into
        shuffled and split data ready to be analyzed (saved in ../processed).'''
        
    # make directories for train, val, and test
    train_dir = os.path.join(output_filepath,"train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_filepath, 'val')
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_filepath, 'test')
    os.makedirs(test_dir, exist_ok=True)

    #Read the images and label from interim folder
    images = glob.glob(input_filepath + '/*.jpg')
    label_path = os.path.join(input_filepath, 'labels.csv')
    label = pd.read_csv(label_path)
    
    #Set seed and shuffle image list
    random.seed(440)
    random.shuffle(images)
    random.shuffle(images)
    random.shuffle(images)
    
    # divide images into different folders alongside the labels
    split_1 = int(0.8 * len(images))
    split_2 = int(0.95 * len(images))
    
    train_images = images[:split_1]
    val_images  = images [split_1:split_2]
    test_images =  images [split_2:]
    
    for image_path in train_images:
        image_name = os.path.basename(image_path)
        final_path = os.path.join(train_dir, image_name)
        shutil.copy(image_path, final_path)
    train_images_names = [os.path.basename(x) for x in train_images]
    train_label = label[label['image'].isin(train_images_names)]
    train_label.to_csv(os.path.join(train_dir, 'label.csv'))
    
    for image_path in val_images:
        image_name = os.path.basename(image_path)
        final_path = os.path.join(val_dir, image_name)
        shutil.copy(image_path, final_path)
    val_images_name = [os.path.basename(x) for x in val_images]
    val_label = label[label['image'].isin(val_images_name)]
    val_label.to_csv(os.path.join(val_dir, 'label.csv'))
    
    for image_path in test_images:
        image_name = os.path.basename(image_path)
        final_path = os.path.join(test_dir, image_name)
        shutil.copy(image_path, final_path)
    test_images_names = [os.path.basename(x) for x in test_images]
    test_label = label[label['image'].isin(test_images_names)]
    test_label.to_csv(os.path.join(test_dir, 'label.csv'))
        
    # logger = logging.getLogger(__name__)
    # logger.info('making interim data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()