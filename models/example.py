# This file demonstrates how to use QD-MSA for multimodal sentiment analysis (MMSA) tasks.

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from common_models import GRU, GRUWithLinear, MMDL, Concat
# Import the split and unsplit quantum models
from quantum_split_model import QNNSplited
from quantum_unsplited_model import QNNUnsplitted


# Define device (use CPU if no CUDA available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset type and task
dataset_type = 'mosei' # 'mosei' or 'mosi' dataset
task = ['acc-7', 7] # Task type and number of output classes

# Define encoder hidden dimensions based on dataset
if dataset_type == 'mosi':
     encoder_hidden = {0:300, 1:300, 2:400}
     # Define encoders for each modality (text, audio, video)
     encoders = [GRUWithLinear(35,100, encoder_hidden[0], dropout=1e-1, has_padding=True, batch_first=True).to(device),
               GRUWithLinear(74,300, encoder_hidden[1], dropout=1e-1, has_padding=True, batch_first=True).to(device),
               GRUWithLinear(300,900, encoder_hidden[2], dropout=1e-1, has_padding=True, batch_first=True).to(device)]
elif dataset_type == 'mosei':
     # text vision audio
     encoder_hidden = {0:600, 1:200, 2:200}
     # Define encoders for each modality (text, vision, audio)
     encoders = [GRUWithLinear(713,500,encoder_hidden[0], dropout=1e-1, has_padding=True, batch_first=True).to(device),
              GRUWithLinear(74,300,encoder_hidden[1], dropout=1e-1, has_padding=True, batch_first=True).to(device),
               GRUWithLinear(300,600,encoder_hidden[2], dropout=1e-1, has_padding=True, batch_first=True).to(device)]
     
# Define which modalities to use for fusion
masks = [0,1,2]
# Define the fusion module (concatenation of selected modalities)
fusion = Concat(masks = masks).to(device)
# Calculate the input shape for the head based on encoder hidden dimensions
inputShape = sum(encoder_hidden[m] for m in masks)

# Define the head module (either QNNSplited or QNNUnsplitted)
# Use the split quantum model
head = QNNSplited(input_shape=inputShape, output_shape=task[1], hidden_dim=512, with_shortcut=True)
# Use the unsplit quantum model (uncomment and comment the above line to use)
# head = QNNUnsplitted(input_shape=inputShape, output_shape=task[1], hidden_dim=512, with_shortcut=True)

# Instantiate the MMDL model with encoders, fusion, and head
QD_MSA = MMDL(encoders, fusion, head, has_padding=True).to(device)
