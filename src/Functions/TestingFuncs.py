import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot as plt
from src.DataPreprocessing import full_signal
import pickle
import pennylane as qml
from jax import numpy as jnp
import os
import pickle


def make_predictions(circuit, weights, inputs):
    """
    Uses test data and weights obtained from training to make predictions.

            Parameters:
                    circuit (function): The vmapped circuit to make predictions.
                    weights (Jax tensor): Array of weights obtained from training.
                    inputs (numpy.array): test data. Shape is (batch_size, n_qubits) where batch_size = train_size//max_steps.
            Returns:
                    predictions: An array of predicted values, each evaluated by a PQC using weights and some input data. Each value is an expectation value from the PQC and is therefore scale from -1 to 1.
    """

    predictions = circuit(weights, inputs)

    return predictions


def calc_MSE(scaled_inputs, scaled_predictions, scaled_targets, bool_scaled):
    """
    Calculates the mean squared error given predictions and target data. Also calculates the forward using the target data.
            Parameters:
                    scaled_inputs: Data scaled for encoding into the circuit. Default range is 0.2 to 0.8.
                    scaled_predictions: Predictions from the PQC in the range of -1 to 1.
                    scaled_targets: Data scaled using the same strategy as scaled_inputs: default range is 0.2 to 0.8.
                    bool_scaled: If False, the MinMaxScaler method "inverse_transform" is used to return the data to their original values before evaluating the MSE.
            Returns:
                    mse: the mean squared error calculated using the either the scaled or real-value predictions and targets, depending on whether bool_scaled is True
                    forward_mse: the mse obtained from a model which predicts each value x_n as the value before it x_n-1.

    """
    if bool_scaled == True:
        mse = MSE(scaled_predictions, scaled_targets)
        forward_mse = forward(scaled_inputs, scaled_targets)
    else:
        inputs = inverse_transform(scaled_inputs)
        predictions = inverse_transform(scaled_predictions.reshape(-1, 1))
        targets = inverse_transform(scaled_targets)
        mse = MSE(predictions, targets)
        forward_mse = forward(inputs, targets)
    return mse, forward_mse


def forward(inputs, targets):
    """
    A model which predicts each value x_n as the value before it x_n-1.

            Parameters:
                    inputs (numpy.array): Data used by the forward to make a prediction. Shape is (batch_size, n_qubits) where batch_size = train_size//max_steps.
                    targets (numpy.array): True results against which the forward's predictions are compared. Shape is (batch_size, 1).
            Returns:
                    forward_mse: the mse obtained from the "forward" model.
    """
    test_size = len(targets)
    predictions = np.zeros(test_size)
    for i in range(test_size):
        predictions[i] = inputs[i][-1]
    forward_mse = MSE(predictions, targets)
    return forward_mse


def inverse_transform(scaled_data, scaler):
    """
    Performs and inverse transform on predictions and targets to return the data to their original values before scaling with MinMaxScaler

            Parameters:
                    scaled_data: data scaled according to the MinMaxScaler.
                    scaler: The MinMaxScaler initialized in training. To be loaded from a dictionary.
            Returns:
                    data: original data before being scaled with MinMaxScaler
    """
    data = scaler.inverse_transform(scaled_data)
    return data


def test(
    circuit,
    scaled_inputs,
    scaled_targets,
    scaler,
    weights,
    save_mse: str = None,
    bool_scaled=False,
):
    """
    Evaluates the performance of the circuit with final weights on the test dataset.

            Parameters:
                    inputs: Test data
                    weights: Weights obtained from training
                    plot_labels: A list containing the x and y axis labels.
                    bool_plot: If True, a plot of the predicted values over the true values will be shown when the code is run as a cell.
                    save_plot: If None, the plot will not be saved. Otherwise, the plot will be saved to the path given as the string.
                    save_mse (str): If None, the mse will not be saved, otherwise, the mse and forward mse will be saved in a directory called 'MSE to the path given with keys 'Circuit MSE' and 'Forward MSE'.
                    bool_scaled: If True, the mse is calculated using the scaled values. If False, the mse is calculated using the original (unscaled) values.
    """
    scaled_predictions = make_predictions(
        circuit=circuit, weights=weights, inputs=scaled_inputs
    )
    mse, forward_mse = calc_MSE(
        scaled_inputs, scaled_predictions, scaled_targets, bool_scaled=bool_scaled
    )
    if save_mse != None:
        if not os.path.exists(save_mse):
            os.makedirs(save_mse)
        MSE = {
            "Circuit MSE": mse,
            "Forward MSE": forward_mse,
        }
        with open(save_mse + "MSE", "wb") as fp:
            pickle.dump(MSE, fp)
    if bool_scaled == True:
        predictions = inverse_transform(scaled_predictions.reshape(-1, 1), scaler)
        targets = inverse_transform(scaled_targets, scaler)
    return predictions, targets, mse, forward_mse


def plot(
    predictions,
    targets,
    indices,
    n_qubits,
    bool_plot: bool = False,
    save_plot: str = None,
    mse=None,
    forward_mse=None,
    plot_labels=["x-axis", "y-axis"],
):
    """
    Creates a plot of the original data, the targets and the predictions.

            Parameters:
                    predictions: An array of predicted values, each evaluated by a circuit using weights and some input data. Each value has been unscaled to be in the range of the original data.
                    targets: The true valuues against which the predictions are compared.
                    indices: gives the index of each prediction and target in the original data.
                    n_qubits (int): number of qubits in the circuit
                    bool_plot (bool): If True, a plot of the predicted values over the true values will be shown when the code is run as a cell.
                    save_plot: If None, the plot will not be saved. Otherwise, the plot will be saved to the path given as the string.
                    mse: The MSE from the circuit's predictions. To be shown on the plot.
                    forward_mse: The MSE from the forward's predictions. To be shown on the plot.
                    plot_labels: A list of two strings giving the plot's axis labels. Should be of the form ['x-axis','y-axis'].
    """

    if bool_plot == True or save_plot != None:
        if type(plot_labels) != list:
            raise ValueError(
                "Invalid type! input must be a list of the form ['x label','y labe]."
            )
        if len(plot_labels) != 2:
            raise ValueError(
                "Invalid length! input must be a list of the form ['x label','y labe]."
            )
        target_indices = indices[-len(targets) :] + n_qubits - 1

        plt.figure()
        plt.plot(full_signal, label="Signal", alpha=0.5)
        plt.scatter(target_indices, targets, label="Targets", alpha=0.5)
        plt.scatter(target_indices, predictions, label="Predictions", alpha=0.5)
        plt.xlabel(plot_labels[0])
        plt.ylabel(plot_labels[1])
        plt.title("Predictions")
        plt.legend()
        if mse != None and forward_mse != None:
            plt.figtext(x=0, y=0, s="MSE=" + str(mse) + ", Forward=" + str(forward_mse))
        if save_plot != None:
            if not os.path.exists(save_plot):
                os.makedirs(save_plot)
            plt.savefig(save_plot + "predictions")
        if bool_plot == True:
            plt.show
    return


def load_results_params(dict_path):
    """
    Loads saved weights, input and target data for testing, scaler for "unscaling" to original data values, indices for unshuffling the testing data and predictions, and n_qubits.

            Parameters:
                    dict_path: Path of the dictionary containing the desired parameters.
            Returns:
                    results_params: the dictionary containing the desired parameters.
    """
    with open(dict_path + "dict", "rb") as fp:
        results_params = pickle.load(fp)
    return results_params
