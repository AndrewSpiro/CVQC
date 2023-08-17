# %%
from Circuits.Ising import initialize_Ising_circuit
from Circuits.MPS import initialize_MPS_circuit
from Functions.TestingFuncs import * 

# %%
dict = load_results_params('Results/FullApple/MPS/dict')
n_qubits = dict.get("n_qubits")
indices = dict.get("indices")
scaler = dict.get("scaler")
weights = dict.get("weights")
inputs = dict.get("inputs")
targets = dict.get("targets")

vcircuit, circuit, _, n_qubits = initialize_MPS_circuit()
# vcircuit, circuit, _, n_qubits = initialize_Ising_circuit()


print(1)
vpredictions, vtargets, vmse, vforward_mse =test(vcircuit, scaled_inputs = inputs, scaled_targets=targets, scaler = scaler, weights = weights, bool_scaled = True, vmapped = True)
# %%
plot(vpredictions, vtargets, indices, n_qubits, bool_plot = True, save_plot='Results/FullApple/MPS/predictions', mse = vmse, forward_mse=vforward_mse, plot_labels=['Day','Percent of Change'])

# %%
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# import jax
# # %%
# num_initials = 1000
# scaled_results = np.zeros(num_initials)
# results = np.zeros(num_initials)
# n_layers = 2

# for i in tqdm(range(num_initials)):
#     np.random.seed(i)
#     weights = 2 * np.pi * np.random.random(size=(n_layers, 3, n_qubits - 1))
#     #weights = dict.get("weights")
#     # x_init = scaler.transform((60* np.ones(n_qubits-1)).reshape(-1,1))
#     x = 2 * np.pi * np.ones(n_qubits-1)
#     # x = x_init.reshape(1,-1)
#     # x = 60 * np.ones(n_qubits-1)
#     jcircuit = jax.jit(circuit)
#     # scaled_results[i] = (jcircuit(weights, x))
#     results[i] = (jcircuit(weights, x))
    
# # scaled_results = scaled_results.reshape(-1,1)
# # results = inverse_transform(scaled_results, scaler) 
# plt.hist(results)
# %%
