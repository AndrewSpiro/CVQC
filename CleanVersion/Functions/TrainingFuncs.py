import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
from DataPreprocessing import r


def split_train_test(data, n_qubits, train_ratio = 2/3, random = False, seed = 0):
    
    grouped_data = np.zeros((len(data)-n_qubits+1, n_qubits))
    for i in range(len(data)-n_qubits+1):
        grouped_data[i] = (np.array(data[i:i+n_qubits])).squeeze()

    
    shuffle_indices = np.arange(0,len(grouped_data))   
    if random == True:
        np.random.seed(seed)
        np.random.shuffle(shuffle_indices)
    shuffled_grouped_data = grouped_data[shuffle_indices]
    
    train = shuffled_grouped_data[:int(train_ratio*(len(shuffled_grouped_data)))]
    test = shuffled_grouped_data[int(train_ratio*(len(shuffled_grouped_data))):]
    train_size = len(train)
    test_size = len(test)
    
    return train, test, train_size, test_size
    
def scale_data(train, test, train_size, test_size, n_qubits, scaler_min = 0.2, scaler_max = 0.8):
    scaler = MinMaxScaler((scaler_min,scaler_max))
    train_1d = train.reshape(train_size*n_qubits,1)
    test_1d = test.reshape(test_size*n_qubits,1)
    train_test_1d = np.concatenate((train_1d, test_1d))
    scaler.fit(train_test_1d)
    scaled_train_1d = scaler.transform(train_1d)
    scaled_test_1d = scaler.transform(test_1d)
    scaled_train = scaled_train_1d.reshape(train_size,n_qubits)
    scaled_test = scaled_test_1d.reshape(test_size, n_qubits)
    
    final_train = scaled_train
    final_test = scaled_test
    
    return final_train, final_test

def input_target_split(data):
    data_size = len(data)
    x = np.zeros((data_size, r))
    target_y = np.zeros((data_size,1))
    for i in range(data_size):
        x[i] = data[i][:-1]
        target_y[i] = data[i][-1]
    return x, target_y

def train_model(train, test, weights, circuit, max_steps, epochs, loss_function = 'square_loss', optimizer = 'qml.AdamOptimizer' , learning_rate = 0.1, bool_plot = False, save_plot: str = None):
    
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
    np.save(filename, weights)

def save_test_data(x_t, y_targets_t, input_filename: str = "test inputs- rename asap", target_filename: str = "test targets- rename asap"):
    np.save(input_filename, x_t)
    np.save(target_filename, y_targets_t)