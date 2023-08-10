# %%
from Circuits.Ising import initialize_Ising_circuit
from DataPreprocessing import full_signal
from Functions.TrainingFuncs import *

circuit, weights, n_qubits = initialize_Ising_circuit(bool_draw = True)

train, test, train_size, test_size, train_ratio, indices = split_train_test(full_signal, n_qubits, random = True)
final_train, final_test, scaler = scale_data(train, test, train_size, test_size, n_qubits)
# %%
weights, x_t, target_y_t = train_model(final_train, final_test, weights, circuit, n_qubits, max_steps = 10, epochs = 10, bool_plot=True, save_plot='Results/TEST/TEST loss')
# %%
results_and_params = {
    "n_qubits" : n_qubits,
    "indices" : indices,
    "scaler" : scaler,
    "weights" : weights,
    "inputs" : x_t,
    "targets" : target_y_t
}

#save_circuit(circuit, 'Results/TEST/TEST circuit')
save_results_params(results_and_params, 'Results/TEST/TEST dict')
# %%