# %%
from Circuits.Ising import initialize_Ising_circuit
from Functions.TestingFuncs import * 

# %%
dict = load_results_params('Results/TEST/TEST dict')
n_qubits = dict.get("n_qubits")
indices = dict.get("indices")
scaler = dict.get("scaler")
weights = dict.get("weights")
inputs = dict.get("inputs")
targets = dict.get("targets")

circuit, _, n_qubits = initialize_Ising_circuit(bool_draw = True)

#circuit = load_circuit('Results/TEST/TEST circuit')
# %%
predictions, targets, mse, forward_mse =test(circuit, scaled_inputs = inputs, scaled_targets=targets, scaler = scaler, weights = weights, bool_scaled = True)
plot(predictions, targets, indices, n_qubits, bool_plot = True, save_plot= 'Results/TEST/TEST predictions', mse = mse, forward_mse=forward_mse, plot_labels=['Day','Percent of Change'])
# %%