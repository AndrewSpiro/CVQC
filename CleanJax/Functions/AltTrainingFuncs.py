import jax
from jax import numpy as jnp

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
                    indices: gives the index of each "group" in the original data. If random is False, indices is just the range of integers from 0 to the number of groups.
    '''
    
    grouped_data = np.zeros((len(data)-n_qubits+1, n_qubits))
    for i in range(len(data)-n_qubits+1):
        grouped_data[i] = (np.array(data[i:i+n_qubits])).squeeze()

    
    indices = np.arange(0,len(grouped_data))   
    if random == True:
        np.random.seed(seed)
        np.random.shuffle(indices)
    shuffled_grouped_data = grouped_data[indices]
    
    train = shuffled_grouped_data[:int(train_ratio*(len(shuffled_grouped_data)))]
    test = shuffled_grouped_data[int(train_ratio*(len(shuffled_grouped_data))):]
    train_size = len(train)
    test_size = len(test)
    
    return train, test, train_size, test_size, train_ratio, indices

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

def split_input_target(data, n_qubits):
    '''
    For a 2-D array, moves along axis = 0 and splits each array of size N into an "input" array of size N-1 and a "target" array of size N.
    
            Parameters:
                    data (numpy.ndarray): The 2-D array to be split
            Returns:
                    x (numpy.ndarray): A 2-D "input" array. x[i] is a 1-D array containing data[i][:-1]
                    target_y (numpy.ndarray): A 2-D "target" array. target_y[i]  is a 1-D array of containing data[i][-1]
    '''
    data_size = len(data)
    x = jnp.zeros((data_size, n_qubits - 1))
    target_y = jnp.zeros((data_size,1))
    for i in range(data_size):
        x[i] = data[i][:-1]
        target_y[i] = data[i][-1]
    return x, target_y

def data_stream(x, y, num_batches, batch_size):
    for i in range(num_batches):
        inputs, targets = x[i * batch_size : (i+1)*batch_size], y[i * batch_size : (i+1)*batch_size]
    return inputs, targets

def train_step(weights, batch):
    inputs, targets = batch
    out = circuit(weights, inputs) # minmax scaler to go from -1:1 to 0.2:0.8?

def train_model(data, n_qubits,batch_size = 10, n_epochs = 10, learning_rate = 0.1):
    
    train, test, scaler = split_train_test(data, n_qubits)
    train_x, train_y = split_input_target(train, n_qubits)
    test_x, test_y = split_input_target(test, n_qubits)
    
    num_train = len(train_y)
    num_test = len(test_y)
    num_complete_batches, leftover  = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    num_batches_complete_test,leftover = divmod(num_test, batch_size)
    num_batches_test = num_batches_complete_test + bool(leftover)

    batches = data_stream(train_x, train_y, num_batches, batch_size)
    test_batches = data_stream(test_x, test_y, num_batches_test, batch_size)
    

