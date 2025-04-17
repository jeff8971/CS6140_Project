#!/usr/bin/env python3
"""
Filename: csv_data_processing.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script performs multiple functions:
             1. Separates the emotion and pixel data from the train.csv and saves them as emotion.csv and pixels.csv.
             2. Loads pixel data from a CSV file and saves it as individual JPEG images.
             3. Processes facial expression data from a CSV file, segregating images into training and validation datasets
                based on their usage tag, creating directories for each emotion category and saving the images accordingly.
Date: 2025-04-09
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Path Definitions
train_csv_path = 'dataset/csv/src_csv/train.csv'  # Path to the input training CSV file
output_directory = 'dataset/csv/rst_csv'          # Directory to store separated CSV files
csv_path = './dataset/csv/rst_csv/pixels.csv'     # Path to the pixels CSV file
images_path = './dataset/face_images'             # Directory to store generated face images
train_path = 'dataset/img/train/'                 # Directory for training images
valid_path = 'dataset/img/test/'                  # Directory for validation images
data_path = 'dataset/csv/src_csv/fer2013.csv'     # Path to the FER2013 dataset CSV file


def separate_data(csv_path, output_dir):
    """
    Separates emotion and pixel data from the input CSV file into two separate CSV files.
    
    Args:
        csv_path (str): Path to the input CSV file containing both emotion and pixel data
        output_dir (str): Directory where the separated CSV files will be saved
    """
    df = pd.read_csv(csv_path)
    df_y = df[['emotion']]
    df_x = df[['pixels']]
    df_y.to_csv(f'{output_dir}/emotion.csv', index=False, header=False)
    df_x.to_csv(f'{output_dir}/pixels.csv', index=False, header=False)


def save_images_from_csv(csv_file_path, output_path):
    """
    Converts pixel data from CSV into JPEG images and saves them to the specified directory.
    
    Args:
        csv_file_path (str): Path to the CSV file containing pixel data
        output_path (str): Directory where the generated images will be saved
    """
    os.makedirs(output_path, exist_ok=True)
    data = np.loadtxt(csv_file_path)
    for i in range(data.shape[0]):
        face_array = data[i].reshape((48, 48))
        file_path = os.path.join(output_path, f'{i}.jpg')
        cv2.imwrite(file_path, face_array)


def create_directories(base_path):
    """
    Creates subdirectories for each emotion category (0-6) in the specified base path.
    
    Args:
        base_path (str): Base directory path where emotion subdirectories will be created
    """
    for emotion in range(7):
        os.makedirs(os.path.join(base_path, str(emotion)), exist_ok=True)


def process_and_save_images():
    """
    Processes the FER2013 dataset, creating training and validation sets by:
    1. Reading the dataset CSV file
    2. Converting pixel data to images
    3. Organizing images into emotion-specific directories
    4. Separating images into training and validation sets based on usage tag
    """
    df = pd.read_csv(data_path)
    file_counters = {emotion: 1 for emotion in range(7)}  # Counter for each emotion category
    for index, row in df.iterrows():
        emotion, pixels, usage = row['emotion'], row['pixels'], row['Usage']
        image_array = np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48)
        image = Image.fromarray(image_array).convert('L')
        directory = train_path if usage == 'Training' else valid_path
        file_path = os.path.join(directory, str(emotion), f'{file_counters[emotion]}.jpg')
        file_counters[emotion] += 1
        image.save(file_path)


def main():
    """
    Main function that orchestrates the data processing pipeline:
    1. Separates emotion and pixel data
    2. Saves pixel data as images
    3. Creates necessary directories
    4. Processes and saves images into training and validation sets
    """
    separate_data(train_csv_path, output_directory)
    save_images_from_csv(csv_path, images_path)
    create_directories(train_path)
    create_directories(valid_path)
    process_and_save_images()

if __name__ == "__main__":
    main()
