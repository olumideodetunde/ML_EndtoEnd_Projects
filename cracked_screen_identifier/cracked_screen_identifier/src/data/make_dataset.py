# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
import glob
import cv2
import pandas as pd

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        same sized image and generates labels for all the images 
        ready to be used (saved in ../interim).
    """
    
    counter = 0
    images = []
    labels = []
    subdirs = glob.glob(input_filepath + '/*screen') #identify the subdirectories in the input filepath

    for subdir in subdirs:
        label = os.path.basename(subdir).split('_')[0]

        for file_path in glob.glob(subdir + '/*.jpg'):
            img = cv2.imread(file_path)
            new_filename = f'image_{counter}.jpg'
            new_filepath = os.path.join(output_filepath, new_filename)
            cv2.imwrite(new_filepath, img)
            images.append(new_filename)
            labels.append(label)
            counter += 1

    label = pd.DataFrame({'image': images, 'label': labels})
    label.to_csv(os.path.join(output_filepath, 'labels.csv'), index=False)
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
