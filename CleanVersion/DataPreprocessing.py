# %%
from DataPreprocessingFuncs import *
# %%
dataset = load_data('AAPL max.csv',usecols=['Close'], sample_size=201, bool_plot=True)
percent_of_change = gradient(dataset, bool_plot=True)
DC, amp, N = FT_mod_squared(percent_of_change, 1, True, save_components='Apple 200 1')
full_signal, r = build_signal(DC, amp, 0, 0, N, True)
# %%
