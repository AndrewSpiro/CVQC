from Functions.TestingFuncs import *

weights = np.load('Results/TEST/TEST weights.npy')
inputs = np.load('Results/TEST/TEST inputs.npy')
targets = np.load('Results/TEST/TEST targets.npy')

predictions = make_predictions(weights, inputs)
