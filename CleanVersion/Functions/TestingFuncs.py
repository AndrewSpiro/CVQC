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
                    predictions: An array of predicted values, each evaluated by a PQC using weights and some input data. Each value is an expectation value from the PQC and is therefore scale from -1 to 1.
    '''
    predictions = np.zeros((test_size,1))
    for i in range(test_size):
        predictions[i] = (circuit(weights, inputs[i]))
    predictions = predictions.reshape((test_size, 1))
    return predictions


def calc_MSE(scaled_inputs, scaled_predictions, scaled_targets, bool_scaled):
    '''
    Calculates the mean squared error given predictions and target data. Also calculates the forward using the target data.
            Parameters:
                    scaled_predictions: Predictions from the PQC in the range of -1 to 1.
                    scaled_targets: Targets that have been scaled according to the initalization of the MinMaxScaler.
                    bool_scaled: If False, the MinMaxScaler method "inverse_transform" is used to return the data to their original values before evaluating the MSE.
            Returns:
                    mse: the mean squared error calculated using the either the scaled or real-value predictions and targets, depending on whether bool_scaled is True
                    forward_mse: the mse obtained from a model which predicts each value x_n as the value before it x_n-1.

    '''
    if bool_scaled == True:
        mse = mse(scaled_predictions, scaled_targets)
        forward_mse = forward(scaled_inputs, scaled_targets)        
    else:
        inputs = inverse_transform(scaled_inputs)
        predictions = inverse_transform(scaled_predictions)
        targets = inverse_transform(scaled_targets)
        mse = mse(predictions, targets)
        forward_mse = forward(inputs, targets)
    return mse, forward_mse


def forward(inputs, targets):
    '''
    A model which predicts each value x_n as the value before it x_n-1.

            Parameters:
                    x: The data on which predictions are to be made.
            Returns:
                    forward_mse: the mse obtained from the "forward" model.
    '''
    test_size = len(targets)
    predictions = np.zeros(test_size)
    for i in range(test_size):
        predictions[i] = inputs[i][-1]
    forward_mse = metrics.mean_squared_error(predictions,targets)
    print(predictions)
    return forward_mse


def inverse_transform(scaled_data):
    '''
    Performs and inverse transform on predictions and targets to return the data to their original values before scaling with MinMaxScaler

            Parameters:
                    scaled_data: data scaled according to the MinMaxScaler.
            Returns:
                    data: original data before being scaled with MinMaxScaler
    '''
    data = scaler.inverse_transform(scaled_data)
    return data
    
def test(inputs, weights, save_mse: str = None, bool_scaled = False):
    '''
            Parameters:
                    inputs: Test data
                    weights: Weights obtained from training
                    plot_labels: A list containing the x and y axis labels.
                    bool_plot: If True, a plot of the predicted values over the true values will be shown when the code is run as a cell.
                    save_plot: If None, the plot will not be saved. Otherwise, the plot will be saved to the path given as the string.
                    save_mse (str): If None, the mse will not be saved, otherwise, the forward mse will be saved to the path given as the string.
                    bool_scaled: If True, the mse is calculated using the scaled values. If False, the mse is calculated using the original (unscaled) values.
    '''
    scaled_predictions = make_predictions(weights, inputs)
    mse, forward_mse = calc_MSE(scaled_predictions, target_y_t, bool_scaled)
    if bool_scaled == True:
        predictions = inverse_transform(scaled_predictions)
        targets = inverse_transform(target_y_t)
    return predictions, targets, mse, forward_mse
        

def plot(predictions, targets, mse = None, forward_mse = None, plot_labels = ['x-axis','y-axis'], bool_plot: bool = False, save_plot: str = None):
        
    def unshuffle_predictions(targets, indices, train_size):
        sorted_predictions = np.zeros(N)
        for i in range(train_size,len(indices)):
            final_pos = indices[i] + r
            sorted_predictions[final_pos] = targets[i-train_size]
                
        pred_index = []
        pred = []
        for i in range(N):
            if sorted_predictions[i] ==0:
                continue
            pred_index.append([i])
            pred.append([sorted_predictions[i]])
        return pred_index, pred    

    pred_index, pred = unshuffle_predictions

    if bool_plot == True or save_plot != None:
        if type(plot_labels) != list:
            raise ValueError("Invalid type! input must be a list of the form ['x label','y labe].")
        if len(plot_labels) != 2:
            raise ValueError("Invalid length! input must be a list of the form ['x label','y labe].")
        plt.figure()
        plt.plot(full_signal, label = "Signal")
        plt.scatter(pred_index, predictions, label = "Predictions")
        plt.scatter(pred_index, targets, label = "Targets")
        plt.xlabel(plot_labels[0])
        plt.ylabel(plot_labels[1])
        plt.title("Predictions")
        plt.legend()
        if mse != None and forward_mse != None:
            plt.figtext(x=0, y = 0, s = 'MSE=' + str(mse) + ', Forward=' + str(forward_mse))
        if save_plot != None:
            plt.savefig(save_plot)
        if bool_plot == True:
            plt.show
    return