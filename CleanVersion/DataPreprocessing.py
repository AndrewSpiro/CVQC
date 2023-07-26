# %%
from DataPreprocessingFuncs import *
# %%
dataset = load_data('Datasets/AAPLmax.csv',usecols=['Close'], sample_size=201)
percent_of_change = gradient(dataset)
DC, amp, N = FT_mod_squared(percent_of_change, 1)
full_signal, r = build_signal(DC, amp, 0, 0, N)
# %%
