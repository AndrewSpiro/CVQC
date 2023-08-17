# %%
from Circuits.Ising import initialize_Ising_circuit
from Circuits.MPS import initialize_MPS_circuit
from DataPreprocessing import full_signal
from Functions.TrainingFuncs import *
#from Functions.TestingFuncs import load_results_params

print(len(full_signal))
circuit, _,weights, n_qubits = initialize_MPS_circuit(bool_test=True,bool_draw=True)
#circuit, _,weights, n_qubits = initialize_Ising_circuit(bool_test=True,bool_draw=True)


train, test, train_size, test_size, train_ratio, indices = split_train_test(full_signal, n_qubits, random = True)
final_train, final_test, scaler = scale_data(train, test, train_size, test_size, n_qubits)
# %%
weights, x_t, target_y_t = train_model(final_train, final_test, weights, circuit, n_qubits, max_steps = 10, epochs = 300, bool_plot=True, save_plot='Results/FullApple/MPS/loss', learning_rate=0.1)
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
save_results_params(results_and_params, 'Results/FullApple/MPS/dict')
# %%