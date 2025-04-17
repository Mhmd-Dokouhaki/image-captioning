# Image Captioning with CNN-RNN Architecture

## Overview

This repository contains a complete implementation of an image captioning system using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The system takes images as input and generates natural language descriptions of their content.

Image captioning sits at the intersection of computer vision and natural language processing, combining feature extraction from images with sequence modeling for text generation. This implementation follows the encoder-decoder architecture:

1. An **encoder** (CNN) extracts visual features from input images
2. A **decoder** (RNN/LSTM/GRU) generates captions word-by-word based on these features

## Project Structure

image_captioning_assignment/
├── 📁 data/
│ └── 📄 download_flickr.py # Script to download and prepare Flickr8k dataset
├── 📁 models/
│ ├── 📄 encoder.py # CNN encoder implementations (ResNet, MobileNet)
│ ├── 📄 decoder.py # RNN decoder implementations (LSTM, GRU)
│ └── 📄 caption_model.py # Combined encoder-decoder model
├── 📁 utils/
│ ├── 📄 dataset.py # Dataset and data loader utilities
│ ├── 📄 vocabulary.py # Vocabulary building and text processing
│ ├── 📄 trainer.py # Training loop and optimization
│ └── 📄 metrics.py # Evaluation metrics (BLEU, etc.)
├── 📁 notebooks/
│ ├── 📘 1_Data_Exploration.ipynb # Dataset exploration and analysis
│ ├── 📘 2_Feature_Extraction.ipynb # CNN feature extraction
│ ├── 📘 3_Model_Training.ipynb # Model training and monitoring
│ └── 📘 4_Evaluation_Visualization.ipynb# Results analysis and visualization
├── 📄 requirements.txt # Project dependencies
└── 📄 README.md # Project documentation

## Installation

1. Clone this repository:
git clone https://github.com/Mound21k/image-captioning.git
cd image-captioning

2. Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Download the Flickr8k dataset:
python data/download_flickr.py --data_dir ./data

## Dataset

This project uses the Flickr8k dataset, which contains:
- Approximately 8,000 images
- 5 different captions for each image (40,000 captions total)
- A diverse range of scenes, objects, and actions

The `download_flickr.py` script handles:
- Downloading the images and captions
- Preprocessing captions (cleaning, normalization)
- Creating train/validation/test splits
- Organizing files in the expected directory structure

## Usage

The project is organized as a sequence of steps, each implemented in a dedicated notebook:

### 1. Data Exploration

Run the first notebook to understand the dataset characteristics:

* jupyter notebook notebooks/1_Data_Exploration.ipynb

This notebook:
- Explores the image and caption distributions
- Analyzes caption lengths and vocabulary
- Visualizes sample images with their captions
- Builds and saves the vocabulary

### 2. Feature Extraction

Extract image features using pre-trained CNN models:

* jupyter notebook notebooks/2_Feature_Extraction.ipynb

This notebook:
- Implements feature extraction using different CNN backbones (ResNet18, ResNet50, MobileNetV2)
- Compares models in terms of feature dimensions, extraction speed, and memory requirements
- Analyzes feature distributions and properties
- Saves extracted features to disk for efficient training

### 3. Model Training

Train the image captioning model:

* jupyter notebook notebooks/3_Model_Training.ipynb

This notebook:
- Implements the encoder-decoder architecture
- Sets up the training pipeline with teacher forcing
- Trains the model with appropriate hyperparameters
- Monitors training progress and validation performance
- Saves model checkpoints for later evaluation

### 4. Evaluation and Visualization

Evaluate the trained model and visualize results:

* jupyter notebook notebooks/4_Evaluation_Visualization.ipynb

This notebook:
- Generates captions for test images
- Calculates BLEU scores and other metrics
- Analyzes model performance across different image types
- Provides an interactive demo for generating captions on new images
- Compares different decoding strategies (greedy vs. beam search)

## Implementation Details

### Encoder

The encoder module (`models/encoder.py`) provides several CNN options:
- **ResNet18**: A lightweight model with 512-dimensional features
- **ResNet50**: A deeper model with 2048-dimensional features
- **MobileNetV2**: An efficient model with 1280-dimensional features

Each model is pre-trained on ImageNet and modified to output feature vectors of the desired dimension.

### Decoder

The decoder module (`models/decoder.py`) implements:
- LSTM and GRU variants
- Word embedding layer for caption tokens
- Linear projection to vocabulary size
- Optional beam search decoding for improved caption quality

### Caption Model

The combined model (`models/caption_model.py`) connects the encoder and decoder:
- Uses the encoder to extract image features
- Feeds these features to the decoder as initial state
- Implements caption generation using teacher forcing during training
- Provides both greedy and beam search decoding during inference

## Results

The system achieves competitive performance on the Flickr8k dataset:
- BLEU-1: ~0.60-0.65
- BLEU-4: ~0.20-0.25

Example captions generated by the model:
- "A brown dog runs through the grass"
- "A man in a red shirt is climbing a rock wall"
- "Children playing soccer on a green field"

## Dependencies

Main dependencies include:
- PyTorch (1.7.0+)
- torchvision
- numpy
- matplotlib
- nltk
- h5py
- tqdm

See `requirements.txt` for the complete list.

## Extending the Project

You can extend this project in several ways:
1. **Attention Mechanism**: Implement visual attention to focus on relevant image regions
2. **Transformer Architecture**: Replace the RNN decoder with a Transformer
3. **Larger Datasets**: Use MS COCO or Flickr30k for more training data
4. **Different Metrics**: Implement CIDEr or METEOR for evaluation
5. **Fine-tuning**: Enable fine-tuning of the CNN encoder during training

## Acknowledgments

- The implementation is inspired by the "Show and Tell" paper by Vinyals et al.
- Pre-trained models are provided by torchvision
- Flickr8k dataset from the University of Illinois