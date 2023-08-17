from Circuits.Ising import initialize_Ising_circuit
from Circuits.MPS import initialize_MPS_circuit
from Functions.TestingFuncs import * 

dict_path = 'Results/FullApple/MPS/dict'
predictions_path = 'Results/TEST/nonexisting/'

dict = load_results_params(dict_path)
n_qubits = dict.get("n_qubits")
indices = dict.get("indices")
scaler = dict.get("scaler")
weights = dict.get("weights")
inputs = dict.get("inputs")
targets = dict.get("targets")

circuit, _, _, n_qubits = initialize_MPS_circuit()

predictions, targets, mse, forward_mse =test(circuit, scaled_inputs = inputs, scaled_targets=targets, scaler = scaler, weights = weights, bool_scaled = True, vmapped = True)
plot(predictions, targets, indices, n_qubits, bool_plot = True, save_plot=predictions_path, mse = mse, forward_mse=forward_mse, plot_labels=['Day','Percent of Change'])