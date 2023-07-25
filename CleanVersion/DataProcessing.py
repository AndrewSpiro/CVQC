import pandas as pd
from pennylane import numpy as np
from matplotlib import pyplot as plt
import tqdm
from typing import Union

def load_data(dataset, usecols = all, skiprows = None, sample_size = all, bool_plot = False):
    '''
    Loads a specified subset of a csv file. Documentation source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

            Parameters:
                    dataset (str, path object or file-like object): Any valid string path is acceptable. The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is expected. A local file could be: file://localhost/path/to/table.csv. If you want to pass in a path object, pandas accepts any os.PathLike. By file-like object, we refer to objects with a read() method, such as a file handle (e.g. via builtin open function) or StringIO.
                    usecols (list-like or callable, optional): Return a subset of the columns. If list-like, all elements must either be positional (i.e. integer indices into the document columns) or strings that correspond to column names provided either by the user in names or inferred from the document header row(s). If names are given, the document header row(s) are not taken into account. For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz']. Element order is ignored, so usecols=[0, 1] is the same as [1, 0]. To instantiate a DataFrame from data with element order preserved use pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']] for columns in ['foo', 'bar'] order or pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']] for ['bar', 'foo'] order. If callable, the callable function will be evaluated against the column names, returning names where the callable function evaluates to True. An example of a valid callable argument would be lambda x: x.upper() in ['AAA', 'BBB', 'DDD'].
                    skiprows (list-like, int or callable, optional): Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. If callable, the callable function will be evaluated against the row indices, returning True if the row should be skipped and False otherwise. An example of a valid callable argument would be lambda x: x in [0, 2].
                    sample_size (int): Number of datapoints to include i.e., number of rows except headers.
                    bool_plot (boolean): If True, will plot the data
            Returns: 
                    dataset (np.array): a 1-D array

    '''

    data_frame = pd.read_csv(dataset, usecols=usecols)
    data = pd.DataFrame.to_numpy(data_frame)
    np.concatenate(data)
    sub_data = data[:sample_size]
    if bool_plot == True:
        print("Data Subset: ")
        plt.plot(sub_data)
        plt.show()
    return sub_data
    
    
def gradient(data, bool_plot = False):
    '''
    Calculates the percent of change of an input dataset.
    
            Parameters:
                    sub_data ( 1_D np.array of size N): The sequential data for which the gradient should be calculated.
            Returns:
                    pc (1-D np.array of size N-1): The gradient (percent of change) of the input data. The nth value is the rate of change from xn to xn+1.
    '''
    N = len(data)
    pc = np.zeros((N,1))
    for i in range(N-1):
        pc[i] = (data[i+1]-data[i])/data[i]
    if bool_plot == True:
        print("Percent of Change: ")
        plt.figure()
        plt.scatter(np.linspace(0,N,N),pc,marker ='.')
        plt.show
    return pc,N

def FT_mod_squared(signal, threshold: int, bool_plot: bool = False, save_components: str = None):
    '''
    Performs a discrete Fourier transform and calculates the modulus squared to estimate the signal's power spectral density.
    
            Parameters:
                    signal (1-D np.array): Data for which the power spectral density is to be estimated.
                    threshold (int): The minimum amplitude squared required for a component frequency to be added to the list of deterministic components.
                    bool_plot (bool) = If True, will show the periodogram.
                    save_components (str, optional) = The name of the file to which the deterministic components will be saved. If not specified, the deterministic components will not be saved.
            Returns:
                    DC (1-D np.array): Array containing the frequencies of the component signals with amplitudes above the threshold (the deterministic components).
                    amp (1-D np.array): Array containing the amplitudes of the deterministic components. (DC[i] and amp[i] give the frequency and amplitude respectively of a deterministic component signal.)
                    N = length of the original signal.

    '''
    
    N = len(signal)
    P = np.zeros((N//2,1))
    nu = np.zeros((N//2,1))
    for k in tqdm(range(N//2)):
        sum = 0
    for i in range(N):
        sum += (signal[i] * np.exp(2 * np.pi * i * k * 1j * 1/N))
    P[k] = np.abs(sum)**2
    nu[k] = k/N
    if bool_plot == True:
        plt.loglog(nu, P)

    amp2 = []
    DC = []
    for i in range(N//2):
        if P[i] > threshold:
            amp2.append(P[i])
            DC.append(nu[i])
    amp = np.sqrt(amp2)
    if save_components != None:
        np.save(save_components, np.array(DC,amp))
    return(DC, amp, N)

def build_signal(DC, amp, c_noise: Union[0,1,2,3,4], trend_type: Union[0,1,2], N, bool_plot = False):
    '''
    This constructs a signal from three components: sinusoidal signals, noise, and a long term trend.
    
            Parameters:
                    DC (1-D np.array): An array containing the frequencies of the deterministic components.
                    amp (1-D np.array): An array containing the amplitudes of the deterministic components.
                    c_noise (0, 1, 2, 3, or 4): Noise coefficient to determine the strength off the noise signal: 0 gives no noise, 4 gives a noise signal for which the maximum signal to noise ratio is 1. The coefficients scale the noise linearly.
                    trend_type (0,1,2): 0 -> flat trend: the signal has no long term trend. 1 -> linear trend: the signal has a linear long-term trend with gradient 5e-2. 2 -> quadratic trend: the signal has a quadratic linear trend with gradient 10e-5.
                    N = length of the original signal.
            Returns:
                    full_signal (1-D np.array) = The signal consisting of the deterministic components, additive noise, and a long term trend.
                    r  (int) = The number of deterministic components.
'''
    num_components = len(DC)
    interval = (np.linspace(0,N,N)).reshape((N,1))
    components = np.zeros((N,num_components))
    for i in range(N):
        for j in range(num_components):
            components[i][j] = amp[j] * np.sin(DC[j] * interval[i])  # Computing A*sin(omega*x) for each component and x
    DC_signal = np.sum(components, axis = -1)  # Computing A*sin(omega*x) for each x
    DC_signal = DC_signal.reshape(N,1)
    c_n = [0, 0.2, 0.5, 0.8, 1]  # 5 possible noise coefficients
    noise_scale = np.abs(np.max(DC_signal)-np.min(DC_signal))
    np.random.seed(0)
    noise = np.random.uniform(0,1,N) * noise_scale *(c_n[c_noise])
    noise = noise.reshape((N,1))
    trend = np.array([np.zeros(N), 5e-2 * np.linspace(0,N,N), 2* 5e-5 * np.square(np.linspace(0,N,N))])
    trend = trend.reshape(3,N,1)
    
    full_signal = DC_signal + noise + trend[trend_type]
    if bool_plot == True:
        plt.plot(range(N),full_signal)
    r = num_components
    return full_signal, r
       

