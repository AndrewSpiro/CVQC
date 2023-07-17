# %%
# Imports
import pennylane as qml
from pennylane import numpy as np
# %%
# Data processing: sample_size, num_components, c_n, trend
sample_size = 200
num_components = 3
c_n = 0
trend = 0
# Model build: architecture, layers
# Training: sampling_strategy, max_steps, epochs, optimizer
# Analysis: mse_comparison, loss_comparison

# DATA PROCESSING
# Load data
# Get sample, print and plot some

# Create and show the periodogram

# Get some number of dominant components

# Create the distorted signal

# MODEL
# Architecture
# Layers

# TRAINING
# Given the above parameters, this results in an MSE and a cost list
# If you only want to test e.g., different layers, only run from MODEL down.

# ANALYSIS
# The results are saved in a file containing the values of the parameters and an array with the train costs, test costs, and mse. 
# Processing of the results then happens in another file
results = {
  'sample_size': sample_size,
  'num_components': num_components,
  'c_n': c_n,
  'trend': trend,
  'train_cost': np.array([10,20,30,40]),
  'test_cost': np.array([40,50,60,70]),
  'MSE': 80
}
np.save('test 1', results)
# %%
