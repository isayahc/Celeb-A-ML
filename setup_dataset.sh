#!/bin/bash

# Create directories for the CelebA dataset
mkdir -p datasets/celeba

# Download the CelebA dataset
kaggle datasets download -d jessicali9530/celeba-dataset

# Unzip the dataset into the 'celeba' directory
unzip -d datasets/celeba celeba-dataset.zip
