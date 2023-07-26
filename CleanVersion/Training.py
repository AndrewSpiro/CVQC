# %%
from Circuits.Ising import choose_n_qubits, initialize_Ising_circuit
from DataPreprocessing import full_signal, r
from CleanVersion.Functions.TrainingFuncs import *

n_qubits = choose_n_qubits()

circuit, weights = initialize_Ising_circuit(n_qubits)
train, test, train_size, test_size = split_train_test(full_signal, n_qubits)
final_train, final_test = scale_data(train, test, train_size, test_size, n_qubits)

weights, x_t, target_y_t = train_model(final_train, final_test, weights, circuit, max_steps = 10, epochs = 10, bool_plot=True)

save_weights(weights, 'TEST weights')
save_test_data(x_t,target_y_t,'TEST test inputs', 'TEST test targets')
# %%
