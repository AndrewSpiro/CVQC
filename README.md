# GPU-Optimized Machine Learning Code Repository

This repository contains scripts, Jupyter notebooks, and scripts containing function libraries. This repository is designed to facilitate data preprocessing, model training, and testing for time series forecasting tasks with parametrized quantum circuits. The code is optimized for GPU acceleration using the Jax library.

## Contents

### Data Preprocessing

- `DataPreprocessing.py`: Python script for data preprocessing.
- `DataPreprocessing.ipynb`: Jupyter notebook for data preprocessing.

### Training

- `Training.py`: Python script for training machine learning models.
- `Training.ipynb`: Jupyter notebook for training machine learning models.

### Testing

- `Testing.py`: Python script for testing trained models.
- `Testing.ipynb`: Jupyter notebook for testing trained models.

### Function Libraries

#### DataPreprocessingFuncs

- `DataPreprocessingFuncs.py`: Python script containing functions for data preprocessing.

#### TrainingFuncs

- `TrainingFuncs.py`: Python script containing functions for training machine learning models.

#### TestingFuncs

- `TestingFuncs.py`: Python script containing functions for testing trained models.

### Results

The `results` folder contains subfolders that store the results obtained from the Training and Testing stages. Each subfolder is named according to the dataset, architecture, and hyperparameters used in training and testing. This structure ensures reusability and facilitates interpretation of the outcomes.

## **Getting Started**

1. **Installation:**
    - Clone this repository to your local machine.
    - Ensure you have the necessary dependencies installed, including Jax, Jaxlib and Pennylane.
2. **Dataset Setup:**
    - Prepare your time series dataset in a compatible format: CSV file containing sequential data of interest in one column in consecutive rows.
    - Use the provided data preprocessing utilities to modify your data to suit your specific use cases. Functionalities include calculating gradient, extracting dominant signals, and artificially adding noise and long-term trends.
3. **Model Training and Testing:**
    - Explore the Jupyter notebooks to understand the training process and experiment with different circuit architectures.
    - Customize hyperparameters to explore training and model performance.
4. **Function Libraries:**
    - Utilize the function libraries in the Functions directory to extend the functionality or create your own custom scripts.
5. **Circuit Libraries:**
    - Add new circuit architectures in the Circuits directory.

## Contact

If you have any questions, suggestions, or feedback, please feel free to contact Andrew Spiro at andrew.charles.spiro@cern.ch. 
