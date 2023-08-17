from Functions.DataPreprocessingFuncs import *

dataset = load_data('Datasets/AAPLmax.csv',usecols=['Close'], sample_size=500)
percent_of_change = gradient(dataset)

# Use load_PSD if there exists already a file with the power spectral density. Otherwise this functions calculates it.
DC, amp, N = find_components(load_PSD='Results/TEST/Apple/Apple 200 1.npy')


full_signal, r = build_signal(DC, amp, 0, 0, N, bool_plot=True, labels = ['Title', 'X-axis', 'Y-axis'])