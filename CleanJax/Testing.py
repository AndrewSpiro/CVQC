# %%
from Circuits.Ising import initialize_Ising_circuit
from Functions.TestingFuncs import * 

# %%
dict = load_results_params('Results/TEST/TEST dict JAX')
n_qubits = dict.get("n_qubits")
indices = dict.get("indices")
scaler = dict.get("scaler")
weights = dict.get("weights")
inputs = dict.get("inputs")
targets = dict.get("targets")

vcircuit, circuit, _, n_qubits = initialize_Ising_circuit(bool_draw = True)
print(1)
vpredictions, vtargets, vmse, vforward_mse =test(vcircuit, scaled_inputs = inputs, scaled_targets=targets, scaler = scaler, weights = weights, bool_scaled = True, vmapped = True)
plot(vpredictions, vtargets, indices, n_qubits, bool_plot = True, save_plot=None, mse = vmse, forward_mse=vforward_mse, plot_labels=['Day','Percent of Change'])

#circuit = load_circuit('Results/TEST/TEST circuit')
# %%
print(2)
predictions, targets, mse, forward_mse =test(circuit, scaled_inputs = inputs, scaled_targets=targets, scaler = scaler, weights = weights, bool_scaled = True)
plot(predictions, targets, indices, n_qubits, bool_plot = True, save_plot= None, mse = mse, forward_mse=forward_mse, plot_labels=['Day','Percent of Change'])
# %%