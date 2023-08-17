# %%
from Functions.DataPreprocessingFuncs import *

dataset = load_data('Datasets/AAPLmax.csv',usecols=['Close'], sample_size=200)
percent_of_change = gradient(dataset)
DC, amp, N = find_components(load_PSD="Results/TEST/Apple 200 1.npy")

full_signal, r = build_signal(DC, amp, 0, 0, N,bool_plot=True)  # Add plot labels
print(len(full_signal))
# %%