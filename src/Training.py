from Circuits.Ising import initialize_Ising_circuit
from Circuits.MPS import initialize_MPS_circuit
from DataPreprocessing import full_signal, Data
from Functions.TrainingFuncs import *

# Hyperparameters
Data = Data  # Change the dataset in DataPreprocessing
Architecture = "MPS"
Learning_rate = 0.1
Epochs = 10

path = (
    "Results/"
    + Data
    + "/"
    + Architecture
    + "/"
    + str(Learning_rate)
    + "/"
    + str(Epochs)
    + "/"
)  # Folder is automatically created using above specified parameters. Change path here to specify additional hyperparameters.


initialize_circuit = import_circuit(Architecture)
circuit, weights, n_qubits = initialize_circuit()

train, test, train_size, test_size, train_ratio, indices = split_train_test(
    full_signal, n_qubits, random=True
)
final_train, final_test, scaler = scale_data(
    train, test, train_size, test_size, n_qubits
)

weights, x_t, target_y_t = train_model(
    final_train,
    final_test,
    weights,
    circuit,
    n_qubits,
    max_steps=10,
    epochs=Epochs,
    bool_plot=True,
    save_plot=path,
    learning_rate=Learning_rate,
)

results_and_params = {
    "n_qubits": n_qubits,
    "indices": indices,
    "scaler": scaler,
    "weights": weights,
    "inputs": x_t,
    "targets": target_y_t,
    "initialize_circuit": initialize_circuit,
}

save_results_params(results_and_params, path)
