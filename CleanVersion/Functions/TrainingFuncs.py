import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import pickle
from Circuits.Ising import initialize_Ising_circuit


def scale_data(data, scaler_min = 0.2, scaler_max = 0.8):
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
    scaler = init_scaler.fit(data)
    scaled_data = scaler.transform(data)
    
    return scaled_data, scaler

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
    
    # plt.plot(train[:10,:].T)
    return train, test, train_size, test_size, train_ratio, indices

def train_model(train, test, weights, circuit, n_qubits: int, max_steps: int, epochs: int, loss_function = 'square_loss', optimizer = 'qml.AdamOptimizer' , learning_rate = 0.1, bool_plot = False, save_plot: str = None):
    '''
    Trains the PQC using the "train" dataset and according to the specified hyperparameters. 
    
            Parameters:
                    train (numpy.ndarray): 2-D array containing the scaled training data. Used to train the model and update weights.
                    test (numpy.ndarray): 2-D array containing the scaled testing data. Used to check progress of training.
                    weights: parameters of the PQC updated by the optimizer to change output.
                    circuit (pennylane.qnode.QNode): The circuit which encodes the training data and, according to its parameters, returns an output to be compared with "target" data
                    n_qubits (int): The number of qubits in the quantum circuit
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
    circuit_bis = lambda weights, x : circuit(weights, np.pi*x + np.pi/2)*2
    
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
        x = np.zeros((data_size, n_qubits - 1))
        target_y = np.zeros((data_size,1))
        for i in range(data_size):
            x[i] = data[i][:-1]
            target_y[i] = data[i][-1]
        return x, target_y
    
    valid_loss_functions = ["square_loss"]
    if loss_function not in valid_loss_functions:
        raise ValueError("Invalid loss function! Only 'square_loss' is allowed.")
    elif loss_function == "square_loss":
        # def square_loss(targets, predictions):
        #     loss = 0
        #     for t, p in zip(targets, predictions):
        #         loss += (t - p) ** 2
        #     loss = loss / len(targets)

        #     return 0.5*loss
        
        def square_loss(targets, predictions):
            '''
                    Calculates the calculates the mean squared error between 'targets' and 'predictions'. The JAX NumPy operations automatically perform element-wise computations, enabling efficient GPU execution when used with JIT compilation or vectorized mapping (vmap) in JAX.
                    
                            Parameters:
                                    targets: A JAX array representing the ground truth labels
                                    predictions: A JAX array representing the predicted values
                            Returns: The mean squared error between 'targets' and 'predictions'
            '''
            # print(qml.numpy.tensor(targets[:20]))
            # print(qml.numpy.tensor(predictions[:20]))
            # print(targets.shape)
            # print(predictions.shape)
            # print(qml.numpy.tensor(jnp.mean((targets - predictions) ** 2)))
            return jnp.mean((targets - predictions) ** 2)
        
        # def cost(weights, x, y):
        #     predictions = [circuit(weights, x_) for x_ in x]
        #     return square_loss(y, predictions)
        
        
        def cost(weights, x, y):
            predictions = circuit_bis(weights, x)
            # predictions = (negative_predictions+1)/2 # scaling from range -1:1 to 0:1
            # print(type(predictions))
            # print(type(y))
            # print(square_loss(y, predictions))
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
    
    # print(square_loss([1,2,3],np.array([1,2,3]).T))
    
    train_size = len(train)
    batch_size = train_size//max_steps
    cst = []
    cst.append(cost(weights, x, target_y))
    print('1')
    # print(circuit(weights,x))
    cst_t = []
    cst_t.append(cost(weights, x_t, target_y_t))
    print('2')
    # print(circuit(weights,x_t))
    
    c = cost(weights, x, target_y)
    c = cost(weights, x, target_y)  # Calculating the cost using the whole train data
    # test_val = circuit(weights,x) - target_y
    
    print(np.sqrt(c))
    print(np.sqrt(jnp.mean((target_y - circuit_bis(weights,x)) ** 2)))
    print(circuit_bis(weights,x)[:20])
    _,_,_,_ = initialize_Ising_circuit(bool_draw = True, weights = weights, x = np.pi*x[0]*2)
    _,_,_,_ = initialize_Ising_circuit(bool_draw = True, weights = weights, x = np.pi*x[1]*2)
    print(target_y[:20])
    
    for i in tqdm(range(epochs)):
        for step in tqdm(range(max_steps)):
            # Select batch of data
            batch_index = np.random.randint(0, train_size, batch_size)  # Changing second argument from max_steps to len(x)
            x_batch = x[batch_index]
            y_batch = target_y[batch_index]
            # if step<2:
            #     print(x_batch) # check that there is no sign inversion between x and y
            #     print(y_batch)
            #     print(circuit(weights,x_batch))
                
            jax_x_batch = jnp.array(x_batch)
            jax_y_batch = jnp.array(y_batch)
            # print(cost(weights, jax_x_batch, jax_y_batch)) # check that it decreases
            print('3')
            # print(circuit(weights,jax_x_batch))
            gradient = jax.grad(cost)(weights, jax_x_batch, jax_y_batch)
            #gradient, loss = jax.value_and_grad()
            weights -= learning_rate * gradient         
        c = cost(weights, x, target_y)  # Calculating the cost using the whole train data
        # test_val = circuit(weights,x) - target_y
        
        print(np.sqrt(c))
        print(np.sqrt(jnp.mean((target_y - circuit_bis(weights,x)) ** 2)))
        print(circuit_bis(weights,x)[:20])
        print(target_y[:20])
        
        # qml.draw(circuit(weights,x))
        
        # print(test_val[:20])
        # print(circuit(weights,x))
        print('4')
        # print(circuit(weights,x))
        c_t = cost(weights, x_t, target_y_t)
        print('5')
        # print(circuit(weights,x_t))
        cst.append(c)
        cst_t.append(c_t)
        if i == epochs-1:
            predictions = circuit_bis(weights, x_t)
            plt.figure()
            plt.plot(target_y_t,'g')
            plt.plot([a[-1] for a in x_t],'b')
            plt.plot((predictions+1)/2, 'r')
            print(jnp.mean((target_y_t - predictions) ** 2))
            print(jnp.mean((target_y_t + predictions) ** 2))
            
    
    if bool_plot == True or save_plot != None:
        plt.figure()
        plt.semilogy(range(len(cst)), cst, 'b', label = "train")
        plt.semilogy(cst_t, 'r', label = "test")
        plt.ylabel("Cost")
        plt.xlabel("Step")
        plt.title("Loss")
        plt.legend()
        plt.figtext(x=0, y = 0, s = "initial train cost" + str(cst[0]) + "; final train cost:" + str(cst[-1]))
        if save_plot != None:
            plt.savefig(save_plot)
        if bool_plot == True:
            plt.show()
    return(weights, x_t, target_y_t)

def save_results_params(results_and_params, dict_path):
    with open(dict_path, 'wb') as fp:
        pickle.dump(results_and_params, fp)