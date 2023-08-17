# %%
import numpy as np
from matplotlib import pyplot as plt

mse_list = np.load('adam_50_mse.npy')

# %%
plt.hist(mse_list)
# %%
plt.savefig("adam_50.png")

plt.show()
# %%