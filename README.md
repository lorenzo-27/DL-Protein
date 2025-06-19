# Protein Secondary Structure Prediction

This project implements several deep learning models for Protein Secondary Structure Prediction (PSSP), based on the work of [Zhou et al. (2014)](https://arxiv.org/abs/1403.1347).  
The goal is to compare the effectiveness of different neural network architectures, adapting FCN ([Long et al. 2015](https://arxiv.org/abs/1411.4038)) and U-Net ([Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)) networks to the specific domain of protein structure prediction.

## Project Overview

The project implements three different neural architectures to address the PSSP problem:
1. 1D FCN
2. Base 1D U-Net
3. Optimized 1D U-Net

## Dataset and Preprocessing

The project uses two main datasets described by Zhou et al. (2014):

- **CullPDB 6133 filtered**: training dataset containing 6133 proteins
    
- **CB513**: test dataset containing 513 proteins
    

Both datasets can be downloaded from the [University of Warsaw website](https://lbs.cent.uw.edu.pl/pipred).

### Data Features

Each protein in the dataset includes:

- One-hot encoding of amino acids (20 features)
    
- PSSM (Position-Specific Scoring Matrix) values (20 features)
    
- Positional encoding (1 feature)
    
- Labels for 8 secondary structure classes (Q8)
    

### Preprocessing

- Normalization of PSSM values per protein
    
- Addition of normalized positional encoding
    
- Creation of masks to handle variable-length sequences
    
- Data transposition to optimize convolutional operations
    

## Implemented Models

### 1. 1D FCN

One-dimensional implementation of the FCN described by Long et al. (2015):

- VGG16-style encoder adapted to 1D
    
- Fully connected layers converted into convolutional layers
    
- Multi-scale prediction with feature map fusion
    
- Batch Normalization and ReLU after each convolutional layer
    

### 2. Base 1D U-Net

One-dimensional adaptation of the original U-Net:

- Symmetric encoder-decoder architecture
    
- Skip connections between encoder and decoder
    
- Batch Normalization and ReLU after each convolutional layer
    
- Max pooling and up-sampling operations
    
- Feature map concatenation through skip connections
    

### 3. Optimized 1D U-Net

An improved version of U-Net with optimizations specific to PSSP. It includes all features from the base 1D U-Net, with the addition of:

- Initial Batch Normalization on the input
    
- Dropout in each convolutional layer
    
- He Kaiming weight initialization
    
- Residual connections within convolutional blocks
    

## Training

### Configuration

- Loss Function: Cross Entropy Loss with masking
    
- Optimizer: AdamW with weight decay
    
- Learning Rate: Adaptive scheduling with ReduceLROnPlateau
    
- Early Stopping: Monitoring loss on the CB513 dataset
    
- Batch Size: Configurable via YAML file
    
- Gradient Clipping: Maximum norm 1.0
    

### Metrics

- Training set loss
    
- Q8 accuracy (8 classes)
    
- Q3 accuracy (3 classes, optional)
    
- Monitoring via TensorBoard
    

## Configuration

Each model has an associated YAML configuration file that defines:

- Architecture parameters
    
- Training parameters
    
- Checkpoint directories
    
- Logging and saving frequency
    
- Dataset locations
    

## Requirements

You can install the required packages with:

```bash
pip3 install -r requirements.txt
```

## Usage

```bash
# Train a model
python3 train.py models/model_config.yaml

# Visualize results
tensorboard --logdir tensorboard/model/
```

## Project Structure

```
.
├── checkpoints/
├── data/
│   ├── cullpdb+profile_6133.npy.gz
│   └── cb513+profile_split1.npy.gz
├── models/
│   ├── m_fcn.py
│   ├── m_unet.py
│   ├── m_unet_optimized.py
│   ├── m_fcn.yaml
│   ├── m_unet.yaml
│   └── m_unet_optimized.yaml
├── tensorboard/
│   ├── fcn/
│   ├── unet/
│   └── unet_optimized/
├── dataset.py
├── train.py
└── visualize_network.py
```
