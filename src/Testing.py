from Functions.TestingFuncs import *

# Hyperparameters
Data = "AAPLmax"  # Provide as a string csv file name without .csv extension
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

dict = load_results_params(path)
n_qubits = dict.get("n_qubits")
indices = dict.get("indices")
scaler = dict.get("scaler")
weights = dict.get("weights")
inputs = dict.get("inputs")
targets = dict.get("targets")
initialize_circuit = dict.get("initialize_circuit")

circuit, _, n_qubits = initialize_circuit()

predictions, targets, mse, forward_mse = test(
    circuit,
    scaled_inputs=inputs,
    scaled_targets=targets,
    scaler=scaler,
    weights=weights,
    save_mse=path,
    bool_scaled=True,
)
plot(
    predictions,
    targets,
    indices,
    n_qubits,
    bool_plot=True,
    save_plot=path,
    mse=mse,
    forward_mse=forward_mse,
    plot_labels=["Day", "Percent of Change"],
)
