import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from DataPreprocessing import N
from Training import *

def make_predictions(weights, inputs):
    '''
    Uses test data and weights obtained from training to make predictions.
            Parameters:
                    weights: Array of weights obtained from training.
                    inputs: test data.
            Returns:
                    predictions: An array of predicted values, each evaluated using weights and some input data.
    '''
    predictions = np.zeros((test_size,1))
    for i in range(test_size):
        predictions[i] = (circuit(weights, inputs[i]))
    predictions = predictions.reshape((test_size, 1))
    return predictions

def inverse_transform(predictions, target):
    real_predictions = scaler.inverse_transform(predictions)
    real_target = scaler.inverse_transform(target)
    return real_predictions, real_target

def unshuffle_predictions(real_predictions):
    unshuffled_pred = np.zeros(N)
    for i in range(len(real_predictions)):
        unshuffled_pred[(shuffle_indices[train_size:][i]) * n_qubits] = real_predictions[i]
            
    pred_index = []
    pred = []
    for i in range(N):
        if unshuffled_pred[i] ==0:
            continue
        pred_index.append([i])
        pred.append([unshuffled_pred[i]])
    return pred_index, pred

def forward(x):
    data = x[:-1]
    predictions = x[1:]
    forward_mse = metrics.mean_squared_error(predictions,data)
    return forward_mse

def get_results(pred_index, pred, real_predictions, real_target, plot_labels, bool_plot: bool = False, save_plot: str = None, save_mse: str = None):
        
    if bool_plot == True or save_plot != None:
        if type(plot_labels) != list:
            raise ValueError("Invalid type! input must be a list of the form ['x label','y labe].")
        if len(plot_labels) != 2:
            raise ValueError("Invalid length! input must be a list of the form ['x label','y labe].")
        plt.figure()
        plt.plot(full_signal, label = "signal")
        plt.scatter(pred_index,pred, label = "predictions")
        plt.xlabel(plot_labels[0])
        plt.ylabel(plot_labels[1])
        plt.title("Predictions")
        plt.legend()
        plt.figtext(x=0, y = 0, s = 'MSE=' + str(mse) + ', Forward=' + str(forward_mse))
        plt.savefig(save_plot)
    return mse, forward_mse
    
def test(inputs, weights, plot_labels = ['x-axis','y-axis'], bool_plot: bool = False, save_plot: str = None, save_mse: str = None, bool_scaled = False):
    '''
            Parameters:
                    inputs: Test data
                    weights: Weights obtained from training
                    plot_labels: A list containing the x and y axis labels.
                    bool_plot: If True, the a plot of the predicted values over the true values will be shown when the code is run as a cell.
                    save_plot: If None, the plot will not be saved. Otherwise, the plot will be saved to the path given as the string.
                    save_mse (str): If None, the mse will not be saved, otherwise, the forward mse will be saved to the path given as the string.
                    bool_scaled: If True, the mse is calculated using the scaled values. If False, the mse is calculated using the original (unscaled) values.
    '''
# scaled_predictions = make predictions()
# get the mse
    # if the scaled_bool == True
    #   mse = mse(scaled_predictions, target_y_t)
    #   forward_mse = forward(target_y_t)        
    # else:
    #   predictions = inverse_transform(scaled_predictions)
    #   target = inverse_transform(target_y_t)
    #   mse = mse(predictions, target)
    #   forward_mse = forward(target)
# if scaled_bool == True:
    #   predictions = inverse_transform(scaled_predictions)
    #   target = inverse_transform(target_y_t)
# if bool_plot == True or save_plot != None:
    #if type(plot_labels) != list:
     #raise ValueError("Invalid type! input must be a list of the form ['x label','y labe].")
    #if len(plot_labels) != 2:
     #raise ValueError("Invalid length! input must be a list of the form ['x label','y labe].")
    # plt.figure()
