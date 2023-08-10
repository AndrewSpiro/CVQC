# %%
from Functions.DataPreprocessingFuncs import *
# %%
dataset = load_data('Datasets/AAPLmax.csv',usecols=['Close'], sample_size=201)
percent_of_change = gradient(dataset)
DC, amp, N = find_components(signal=percent_of_change, threshold = 1, save_components='Results/TEST/Apple 200 1')
full_signal, r = build_signal(DC, amp, 0, 0, N, bool_plot=True)  # Add plot labels
# %%