import pennylane as qml
from CleanVersion.DataPreprocessingFuncs import r

# Hyperparameters of the circuit
n_layers = 2
N_QUBITS = (r + 1)

# Initializing the device
dev = qml.device('lightning.qubit', wires= N_QUBITS)

# Building the block according to Emmanoulopoulos & Dimoska
def block(weights):
    for i in range(1,N_QUBITS):
        qml.IsingXX(weights[0][i-1], wires=[0, i])  # Are the qubits in the right place?
    for i in range(1,N_QUBITS):
        qml.IsingZZ(weights[1][i-1], wires=[0, i])
    for i in range(1,N_QUBITS):
        qml.IsingYY(weights[2][i-1], wires=[0, i])

# Constructing the circuit with angle embedding, the above-defined block, and Pauli-Z for measurement
@qml.qnode(dev, interface="autograd")
def PQC(weights,x):
    qml.AngleEmbedding(x,wires=range(N_QUBITS)[1:])   # Features x are embedded in rotation angles
    for j in range(n_layers):
      block(weights[j])
    return qml.expval(qml.PauliZ(wires=0))