# %%
import numpy as np
from matplotlib import pyplot as plt
mse_list = np.load('adam_20_mse.npy')

# %%
plt.hist(mse_list)
plt.savefig("adam_20")
# %%
