import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
from DataPreprocessing import r


def split_train_test(data, n_qubits: int, train_ratio = 2/3, random = False, seed = 0):
    '''
    Splits data into two sub-datsets: one for training and one for testing. 
    
            Parameters:
                    data: 1-D data to be split
                    n_qubits (int): number of qubits in the PQC
                    train_ratio: fraction of full data to be used for training. Must be less than 1.
                    random (bool): if False (default), the data is split chronologically e.g., the first 2/3 of the data are used for training and the last 1/3 are used for testing. If True, the data is split randomly.
                    seed (Any): seed for np.random
            Returns:
                    train (numpy.ndarray): 2-D array containing the training data. The shape of the data is (train_size//n_qubits, n_qubits)
                    test (numpy.ndarray): 2-D array containing the testing data. The shape of the data is (test_size//n_qubits, n_qubits)
                    train_size: the total number of values in train divided by n_qubits (i.e., the number of n_qubits-sized "groups" in train)
                    test_size: the total number of values in test divided by n_qubits (i.e., the number of n_qubits-sized "groups" in test)
                    train_ratio: fraction of full data to be used for training. Must be less than 1.
                    shuffle_indices: gives the index of each "group" in the original data. If random is False, shuffle_indices is just the range of integers from 0 to the number of groups.
    '''
    
    grouped_data = np.zeros((len(data)-n_qubits+1, n_qubits))
    for i in range(len(data)-n_qubits+1):
        grouped_data[i] = (np.array(data[i:i+n_qubits])).squeeze()

    
    data_indices = np.arange(0,len(grouped_data))   
    if random == True:
        np.random.seed(seed)
        shuffle_indices = np.random.shuffle(data_indices)
    else:
        shuffle_indices = data_indices
    shuffled_grouped_data = grouped_data[shuffle_indices]
    
    train = shuffled_grouped_data[:int(train_ratio*(len(shuffled_grouped_data)))]
    test = shuffled_grouped_data[int(train_ratio*(len(shuffled_grouped_data))):]
    train_size = len(train)
    test_size = len(test)
    
    return train, test, train_size, test_size, train_ratio, shuffle_indices
    
def scale_data(train, test, train_size, test_size, n_qubits, scaler_min = 0.2, scaler_max = 0.8):
    '''
    Scales the data for embedding in the PQC. By default, scales all values to be between 0.2 and 0.8.
    
            Parameters:
                    train (numpy.ndarray): 2-D array containing the training data. The shape of the data is (train_size//n_qubits, n_qubits)
                    test (numpy.ndarray): 2-D array containing the testing data. The shape of the data is (test_size//n_qubits, n_qubits)
                    train_size: the total number of values in train divided by n_qubits (i.e., the number of n_qubits-sized "groups" in train)
                    test_size: the total number of values in test divided by n_qubits (i.e., the number of n_qubits-sized "groups" in test)
                    n_qubits: number of qubits in the PQC
                    scaler_min = the target minimum value for all values in the full dataset
                    scaler_max = the target maximum value for all values in the full dataset
            Returns:
                    final_train (numpy.ndarray): 2-D array containing the scaled training data. The shape of the data is (train_size//n_qubits, n_qubits)
                    final_test (numpy.ndarray): 2-D array containing the scaled testing data. The shape of the data is (test_size//n_qubits, n_qubits)
                    scaler: The initialized and fitted scaler                    
    '''    
    init_scaler = MinMaxScaler((scaler_min,scaler_max))
    train_1d = train.reshape(train_size*n_qubits,1)
    test_1d = test.reshape(test_size*n_qubits,1)
    train_test_1d = np.concatenate((train_1d, test_1d))
    scaler = init_scaler.fit(train_test_1d)
    scaled_train_1d = scaler.transform(train_1d)
    scaled_test_1d = scaler.transform(test_1d)
    scaled_train = scaled_train_1d.reshape(train_size,n_qubits)
    scaled_test = scaled_test_1d.reshape(test_size, n_qubits)
    
    final_train = scaled_train
    final_test = scaled_test
    
    return final_train, final_test, scaler

def input_target_split(data):
    '''
    For a 2-D array, moves along axis = 0 and splits each array of size N into an "input" array of size N-1 and a "target" array of size N.
    
            Parameters:
                    data (numpy.ndarray): The 2-D array to be split
            Returns:
                    x (numpy.ndarray): A 2-D "input" array. x[i] is a 1-D array containing data[i][:-1]
                    target_y (numpy.ndarray): A 2-D "target" array. target_y[i]  is a 1-D array of containing data[i][-1]
    '''
    data_size = len(data)
    x = np.zeros((data_size, r))
    target_y = np.zeros((data_size,1))
    for i in range(data_size):
        x[i] = data[i][:-1]
        target_y[i] = data[i][-1]
    return x, target_y

def train_model(train, test, weights, circuit, max_steps, epochs, loss_function = 'square_loss', optimizer = 'qml.AdamOptimizer' , learning_rate = 0.1, bool_plot = False, save_plot: str = None):
    '''
    Trains the PQC using the "train" dataset and according to the specified hyperparameters. 
    
            Parameters:
                    train (numpy.ndarray): 2-D array containing the scaled training data. Used to train the model and update weights.
                    test (numpy.ndarray): 2-D array containing the scaled testing data. Used to check progress of training.
                    weights: parameters of the PQC updated by the optimizer to change output.
                    circuit: The circuit which encodes the training data and, according to its parameters, returns an output to be compared with "target" data
                    max_steps (int): The number of times the weights should be updated before the entire dataset has been evaluated. 
                    max_epochs (int): The maximum number of times the full dataset should be evaluated.
                    loss_function: the function to evaluate the performance of the model i.e., the function to be minimized
                    optimizer: The algorithm that adjusts the weights in the circuit.
                    learning_rate: the learning rate of the optimizer.
                    bool_plot (bool): If True, will plot the loss as a function of epochs once training is completed
                    save_plot (str): If None, the plot will not be saved, otherwise it will be saved as the string entered
            Returns:
                    weights: the most-recently updated weights that were obtained when training is complete
                    x_t: input values for testing
                    target_y_t: target values for training
    '''
    valid_loss_functions = ["square_loss"]
    if loss_function not in valid_loss_functions:
        raise ValueError("Invalid loss function! Only 'square_loss' is allowed.")
    elif loss_function == "square_loss":
        def square_loss(targets, predictions):
            loss = 0
            for t, p in zip(targets, predictions):
                loss += (t - p) ** 2
            loss = loss / len(targets)

            return 0.5*loss
        
        def cost(weights, x, y):
            predictions = [circuit(weights, x_) for x_ in x]
            return square_loss(y, predictions)
    
    valid_optimizers = ['qml.AdamOptimizer', 'qml.AdagradOptimizer']
    if optimizer not in valid_optimizers:
        raise ValueError("Invalid loss function! Only 'square_loss' is allowed.")
    elif optimizer == 'qml.AdamOptimizer':
        opt = qml.AdamOptimizer(learning_rate)
    elif optimizer == 'qml.AdagradOptimizer':
        opt = qml.AdagradOptimizer(learning_rate)      
        
    x, target_y = input_target_split(train)
    x_t, target_y_t = input_target_split(test)
    
    train_size = len(train)
    batch_size = train_size//max_steps
    cst = [cost(weights, x, target_y)]
    cst_t = [cost(weights, x_t, target_y_t)]
    
    for i in tqdm(range(epochs)):
        for step in range(max_steps):
            # Select batch of data
            batch_index = np.random.randint(0, max_steps, batch_size)
            x_batch = x[batch_index]
            y_batch = target_y[batch_index]
            # Update the weights by one optimizer step
            weights,_,_ = opt.step(cost, weights, x_batch, y_batch)  # Calculating weights using the batches.
        c = cost(weights, x, target_y)  # Calculating the cost using the whole train data
        c_t = cost(weights, x_t, target_y_t)
        cst.append(c)
        cst_t.append(c_t)
    
    if bool_plot == True or save_plot != None:
        plt.figure()
        plt.semilogy(range(len(cst)), cst, 'b', label = "train")
        plt.semilogy(cst_t, 'r', label = "test")
        plt.ylabel("Cost")
        plt.xlabel("Step")
        plt.title("Loss")
        plt.legend()
        plt.figtext(x=0, y = 0, s = "initial train cost" + str(cst[0]) + "; final train cost:" + str(cst[-1]))
        if save_plot == True:
            plt.savefig(save_plot)
        if bool_plot == True:
            plt.show()
    
    return(weights, x_t, target_y_t)

def save_weights(weights, filename: str = "weights- rename asap"):
    '''
    Saves final weights from training
    
            Parameters:
                    weights: the weights to be saved
                    filename: the name of the file to which the weights are saved. If not specified, the weights will be saved by default to "weights- rename asap"
    '''
    np.save(filename, weights)

def save_test_data(x_t, y_targets_t, input_filename: str = "test inputs- rename asap", target_filename: str = "test targets- rename asap"):
    '''
    Saves final test inputs and test targets from training
    
            Parameters:
                    x_t: the test inputs to be saved
                    y_targets_t: the test targets to be saved
                    input_filename: the name of the file to which the test inputs are saved. If not specified, the inputs will be saved by default to "test inputs- rename asap"
                    target_filename: the name of the file to which the test targets are saved. If not specified, the targets will be saved by default to "test targets- rename asap"
    '''
    np.save(input_filename, x_t)
    np.save(target_filename, y_targets_t)