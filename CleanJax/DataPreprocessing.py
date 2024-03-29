# %%
from Functions.DataPreprocessingFuncs import *
# %%
dataset = load_data('Datasets/AAPLmax.csv',usecols=['Close'], sample_size=500)
percent_of_change = gradient(dataset)
DC, amp, N = find_components(load_PSD='Results/TEST/Apple 200 1.npy')
full_signal, r = build_signal(DC, amp, 0, 0, N, bool_plot=True, labels = ['F0', 'Day', 'Percent of change'])  # Add plot labels
# %%