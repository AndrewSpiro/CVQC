# %%
import pennylane as qml
from pennylane import numpy as np
# %%
num_results = 1

results_list = []
results_list.append(np.load('test 1.npy', allow_pickle=True))

# %%
train_costs = np.zeros(num_results)
for results in results_list:
    train_costs[results] = results.item().get('train_cost')

for train_cost in train_costs:
    plt.semilogy(range(len(train_cost)), train_cost, 'b')
    
plt.ylabel("Cost")
plt.xlabel("Step")
plt.show()
# %%
