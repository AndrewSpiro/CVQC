# %%
import pennylane as qml
from pennylane import numpy as np
import numpy
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt 

# %%

data_frame = pd.read_csv("AAPL max.csv", usecols=["Close"])
#data_frame = pd.read_csv("UK inflation all time.csv",usecols =[1],skiprows = 183)

data = pd.DataFrame.to_numpy(data_frame)
np.concatenate(data)
# #sample_size = 10080  # Limiting dataset to what was used in the paper.
sample_size = 200
sub_data = data[:sample_size + 1]  # A small sample

print(sub_data[:10])
plt.plot(sub_data)

# %%    
N = sample_size
pc = np.zeros((N,1))
for i in range(N-1):
  pc[i] = (sub_data[i+1]-sub_data[i])/sub_data[i]
# %%
plt.scatter(np.linspace(0,N,N),pc,marker ='.')
#plt.ylim(-0.6,0.6)
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
    
plt.loglog(nu, P)
# %%
num_components = 3

P_1d = np.concatenate(P)
ind = np.argpartition(P_1d, -num_components)[-num_components:]

amp2 = np.zeros(num_components)
DC = np.zeros(num_components)
for i in range(num_components):
  amp2[i] = P[ind[i]]
  DC[i] = nu[[ind[i]]]
amp = np.sqrt(amp2)
print(amp)
print(DC)

# threshold = 1
# amp2 = []
# DC = []
# for i in range(N//2):
#     if P[i] > threshold:
#       amp2.append(P[i])
#       DC.append(nu[i])
# amp = np.sqrt(amp2)
# print(amp)
# print(DC)

# %%

#num_components = len(DC)
DC_sample = DC[0:num_components]
amp_sample = amp[0:num_components]
interval = (np.linspace(0,N,N)).reshape((N,1))
components = np.zeros((N,num_components))
for i in range(N):
    for j in range(num_components):
        components[i][j] = amp_sample[j] + np.sin(DC_sample[j] * interval[i])
DC_signal = np.sum(components, axis = -1)
DC_signal = DC_signal.reshape(N,1)
#  %%
c_n = [0, 0.2, 0.5, 0.8, 1]
noise_scale = np.abs(np.max(DC_signal)-np.min(DC_signal))
np.random.seed(0)
noise = np.random.uniform(0,1,N) * noise_scale *(c_n[3])
noise = noise.reshape((N,1))
trend = np.array([np.zeros(N), 5e-2 * np.linspace(0,N,N), 2* 5e-5 * np.square(np.linspace(0,N,N))])
trend = trend.reshape(3,N,1)

full_signal = DC_signal + noise + trend[0]
plt.plot(range(N),full_signal)
# %%
r = num_components
#r = 16  # Following the paper
N_QUBITS = (r + 1)
print(N_QUBITS)
n_layers = 2

weights = 2 * np.pi * np.random.random(size=(n_layers, 3, r), requires_grad=True)
x = 2 * np.pi *np.random.random(size = (r))

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

PQC(weights, x)
print(qml.draw(PQC,expansion_strategy ="device")(weights,x))
# %%
split_denom = 3
cut_factor = split_denom * N_QUBITS

trunc_size = int(N/cut_factor)*cut_factor
trunc_signal = full_signal[:trunc_size]
grouped_signal = trunc_signal.reshape(trunc_size//N_QUBITS, N_QUBITS)

shuffle_indices = np.arange(0,len(grouped_signal))
np.random.shuffle(shuffle_indices)
gsh_signal = grouped_signal[shuffle_indices]

train = gsh_signal[:int(2/3*(len(gsh_signal)))]
test = gsh_signal[int(2/3*(len(gsh_signal))):]
train_size = len(train)
test_size = len(test)
# %%
scaler = MinMaxScaler((0.2,0.8))
train_1d = train.reshape(train_size*N_QUBITS,1)
test_1d = test.reshape(test_size*N_QUBITS,1)
scaler.fit(train_1d)
scaled_train_1d = scaler.transform(train_1d)
scaled_test_1d = scaler.transform(test_1d)
scaled_train = scaled_train_1d.reshape(train_size,N_QUBITS)
scaled_test = scaled_test_1d.reshape(test_size, N_QUBITS)

final_train = scaled_train
final_test = scaled_test
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

x = np.zeros((train_size, r))
target_y = np.zeros((train_size,1))
for i in range(train_size):
  x[i] = final_train[i][:-1]
  target_y[i] = final_train[i][-1]
  
x_t = np.zeros((test_size, r))  # Already grouped and scaled
target_y_t = np.zeros((test_size,1))
for i in range(test_size):
  x_t[i] = final_test[i][:-1]
  target_y_t[i] = final_test[i][-1]
  
max_steps = 10
optimizer = [qml.AdamOptimizer(.1), qml.AdagradOptimizer(.1)]
opt = optimizer[1]
batch_size = train_size//max_steps
cst = [cost(weights, x, target_y)]  # initial cost
cst_t = [cost(weights, x_t, target_y_t)]

epochs = 10
# %%
for i in tqdm(range(epochs)):
  for step in range(max_steps):
      # Select batch of data
      batch_index = numpy.random.randint(0, max_steps, batch_size)
      x_batch = x[batch_index]
      y_batch = target_y[batch_index]
      # Update the weights by one optimizer step
      weights,_,_ = opt.step(cost, weights, x_batch, y_batch)  # Calculating weights using the batches.
  c = cost(weights, x, target_y)  # Calculating the cost using the whole train data
  c_t = cost(weights, x_t, target_y_t)
  cst.append(c)
  cst_t.append(c_t)
# %%

plt.semilogy(range(len(cst)), cst, 'b')
plt.semilogy(cst_t, 'r')
plt.ylabel("Cost")
plt.xlabel("Step")
plt.show()
print("initial cost" + str(cst[0]))
print("final cost:" + str(cst[-1]))
# %%
t_predictions = np.zeros((test_size,1))
for i in range(test_size):
  t_predictions[i] = (PQC(weights, x_t[i]))
t_predictions = t_predictions.reshape((test_size, 1))

real_predictions = scaler.inverse_transform(t_predictions)
real_target = scaler.inverse_transform(target_y_t)

plt.axline((1.5, 1.5), slope=1,color = '0',linestyle = '--')
plt.plot(real_target[:-1], real_predictions[1:])
plt.plot(real_target[:-1], real_target[1:])
# %%
unshuffled_pred = np.zeros(N)
for i in range(len(real_predictions)):
  unshuffled_pred[(shuffle_indices[train_size:][i]) * N_QUBITS] = real_predictions[i]

pred_index = []
pred =[]
for i in range(N):
  if unshuffled_pred[i] ==0:
    continue
  pred_index.append([i])
  pred.append([unshuffled_pred[i]])

# %%
plt.plot(full_signal)
plt.scatter(pred_index,pred)
print('MSE: ' + str(metrics.mean_squared_error(real_predictions,real_target)))
