# %%
import pennylane as qml
from pennylane import numpy as np
import numpy
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt

apple = pd.read_csv("AAPL max.csv", usecols=["Close", "Date"])

apl = pd.read_csv("AAPL max.csv", usecols = ["Close"])
apl = pd.DataFrame.to_numpy(apl)
np.concatenate(apl)
sample_size = 201
aapl = apl[:sample_size]  # A small sample
print(apl)
plt.plot(aapl)
    
N = sample_size -1
pc = np.zeros((N,1))
for i in range(N-1):
  pc[i] = (aapl[i+1]-aapl[i])/aapl[i]
# %%
plt.scatter(np.linspace(0,N,N),pc,marker ='.')
plt.show

# %%
P = np.zeros((N//2,1))
nu = np.zeros((N//2,1))
for k in tqdm(range(N//2)):
    sum = 0
    for i in range(N):
        sum += (pc[i] * np.exp(2 * np.pi * i * k * 1j * 1/N))
    P[k] = np.abs(sum)**2
    nu[k] = k/N
 
threshold = 1    
plt.loglog(nu, P)
# %%
amp2 = []
DC = []
for i in range(N//2):
    if P[i] > threshold:
      amp2.append(P[i])
      DC.append(nu[i])
amp = np.sqrt(amp2)
print(amp)
print(DC)

num_components = len(DC)
DC_sample = DC[0:num_components]
amp_sample = amp[0:num_components]
interval = (np.linspace(0,N,N)).reshape((N,1))
components = np.zeros((N,num_components))
for i in range(N):
    for j in range(num_components):
        components[i][j] = amp_sample[j] + np.sin(DC_sample[j] * interval[i])
DC_signal = np.sum(components, axis = -1)
DC_signal = DC_signal.reshape(N,1)

full_signal = DC_signal
plt.plot(range(N),full_signal)
# %%
r = num_components
N_QUBITS = (r + 1)
n_layers = 2

dev = qml.device('default.qubit', wires= N_QUBITS)

def block(weights):
    for i in range(1,N_QUBITS):
        qml.IsingXX(weights[0][i-1], wires=[0, i])  # Are the qubits in the right place?
    for i in range(1,N_QUBITS):
        qml.IsingZZ(weights[1][i-1], wires=[0, i])
    for i in range(1,N_QUBITS):
        qml.IsingYY(weights[2][i-1], wires=[0, i])

@qml.qnode(dev, interface="autograd")
def PQC(weights,x):
    qml.AngleEmbedding(x,wires=range(N_QUBITS)[1:])   # Features x are embedded in rotation angles
    for j in range(n_layers):
      block(weights[j])
    return qml.expval(qml.PauliZ(wires=0))

# %%
train = full_signal[:int(N*2/3)]
test = full_signal[int(N*2/3):]

train_size = (len(train)//N_QUBITS) * N_QUBITS
test_size = (len(test)//N_QUBITS) * N_QUBITS

train = train[:train_size]
test = test[:test_size]

scaler = MinMaxScaler((0.2,0.8))
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

grouped_train = scaled_train.reshape(train_size//N_QUBITS, N_QUBITS)
grouped_test = scaled_test.reshape(test_size//N_QUBITS, N_QUBITS)

final_train = grouped_train
final_test = grouped_test
# %%
def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss += (t - p) ** 2
    loss = loss / len(targets)

    return 0.5*loss

def cost(weights, x, y):
    predictions = [PQC(weights, x_) for x_ in x]
    return square_loss(y, predictions)

x = np.zeros((train_size//N_QUBITS, r))
target_y = np.zeros((train_size//N_QUBITS,1))
for i in range(train_size//N_QUBITS):
  x[i] = final_train[i][:-1]
  target_y[i] = final_train[i][-1]
  
x_t = np.zeros((test_size//N_QUBITS, r))  # Already grouped and scaled
target_y_t = np.zeros((test_size//N_QUBITS,1))
for i in range(test_size//N_QUBITS):
  x_t[i] = final_test[i][:-1]
  target_y_t[i] = final_test[i][-1]
  
max_steps = 10 # increase for larger sample sizes
optimizer = [qml.AdamOptimizer(.1), qml.AdagradOptimizer(.1)]
opt = optimizer[0]
batch_size = train_size//max_steps
# %%
epochs = 10
num_initializations = 20
mse_list = np.zeros(num_initializations)

for i in tqdm(range(num_initializations)):
    np.random.seed(i)
    weights = 2 * np.pi * np.random.random(size=(n_layers, 3, r), requires_grad=True)
    cst = [cost(weights, x, target_y)]  # initial cost
    cst_t = [cost(weights, x_t, target_y_t)]
    
    for j in tqdm(range(epochs)):
        for step in range(max_steps):
            # Select batch of data
            batch_index = numpy.random.randint(0, max_steps, batch_size)
            x_batch = x[batch_index]
            y_batch = target_y[batch_index]
            # Update the weights by one optimizer step
            weights,_,_ = opt.step(cost, weights, x_batch, y_batch)  # Calculating weights using the batches.


    t_predictions = np.zeros((test_size//N_QUBITS,1))
    for j in range(test_size//N_QUBITS):
        t_predictions[j] = (PQC(weights, x_t[j]))
        t_predictions = t_predictions.reshape((test_size//N_QUBITS, 1))

    real_predictions = scaler.inverse_transform(t_predictions)
    real_target = scaler.inverse_transform(target_y_t)

    mse_list[i] = metrics.mean_squared_error(real_predictions,real_target)
    print('MSE: ' + str(mse_list[i]))

# %%
np.save('mse_list.npy',mse_list)