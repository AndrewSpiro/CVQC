# %%
from Circuits.Ising import initialize_Ising_circuit
from DataPreprocessing import full_signal
from Functions.TrainingFuncs import *

circuit, weights, n_qubits = initialize_Ising_circuit(bool_draw = True)

train, test, train_size, test_size, train_ratio, shuffle_indices = split_train_test(full_signal, n_qubits)
final_train, final_test, scaler = scale_data(train, test, train_size, test_size, n_qubits)
weights, x_t, target_y_t = train_model(final_train, final_test, weights, circuit, max_steps = 10, epochs = 10, bool_plot=True)

save_weights(weights, 'Results/TEST/TEST weights')
save_test_data(x_t,target_y_t,'Results/TEST/TEST test inputs', 'Results/TEST/TEST test targets')
# %%