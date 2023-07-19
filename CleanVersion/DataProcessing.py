import pandas as pd
from pennylane import numpy as np
from matplotlib import pyplot as plt
import tqdm

# This gives a dataset which is the percent of change in the data from one time step to the next
def percent_of_change(dataset, usecols, skiprows, sample_size):
    data_frame = pd.read_csv("AAPL max.csv", usecols=["Close"])
    data = pd.DataFrame.to_numpy(data_frame)
    np.concatenate(data)
    sub_data = data[:sample_size + 1]
    print("Data Subset: ")
    plt.plot(sub_data)
    plt.show()
    N = sample_size
    pc = np.zeros((N,1))
    for i in range(N-1):
        pc[i] = (sub_data[i+1]-sub_data[i])/sub_data[i]
    print("Percent of Change: ")
    plt.figure()
    plt.scatter(np.linspace(0,N,N),pc,marker ='.')
    plt.show
    return pc,N

# This gives the power spectral density of the "Percent of Change" dataset
def PSD(pc, N):
    P = np.zeros((N//2,1))
    nu = np.zeros((N//2,1))
    for k in tqdm(range(N//2)):
        sum = 0
    for i in range(N):
        sum += (pc[i] * np.exp(2 * np.pi * i * k * 1j * 1/N))
    P[k] = np.abs(sum)**2
    nu[k] = k/N
    plt.loglog(nu, P)
    return(nu, P)

# This constructs a signal from three components: sinusoidal signals, noise, and a long term trend.
def signal_construction(nu, P, threshold,noise,trend, N):
    amp2 = []
    DC = []
    for i in range(N//2):
        if P[i] > threshold:
            amp2.append(P[i])
            DC.append(nu[i])
    amp = np.sqrt(amp2)
    num_components = len(DC)
    interval = (np.linspace(0,N,N)).reshape((N,1))
    components = np.zeros((N,num_components))
    for i in range(N):
        for j in range(num_components):
            components[i][j] = amp[j] + np.sin(DC[j] * interval[i])  # Computing A*sin(omega*x) for each component and x
    DC_signal = np.sum(components, axis = -1)  # Computing A*sin(omega*x) for each x
    DC_signal = DC_signal.reshape(N,1)
    c_n = [0, 0.2, 0.5, 0.8, 1]  # 5 possible noise coefficients
noise_scale = np.abs(np.max(DC_signal)-np.min(DC_signal))
numpy.random.seed(0)
noise = np.random.uniform(0,1,N) * noise_scale *(c_n[0])
noise = noise.reshape((N,1))
trend = np.array([np.zeros(N), 5e-2 * np.linspace(0,N,N), 2* 5e-5 * np.square(np.linspace(0,N,N))])
trend = trend.reshape(3,N,1)
       

