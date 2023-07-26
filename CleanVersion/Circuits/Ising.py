import pennylane as qml
from pennylane import numpy as np
from DataPreprocessing import r

def initialize_circuit(n_qubits: int = r+1, n_layers: int = 2, bool_test = False, bool_draw = False):
    dev = qml.device('lightning.qubit', wires= n_qubits)
    weights = 2 * np.pi * np.random.random(size=(n_layers, 3, n_qubits - 1), requires_grad=True)
    x = 2 * np.pi *np.random.random(size = (n_qubits-1))
        
    def block(weights):
        for i in range(1,n_qubits):
            qml.IsingXX(weights[0][i-1], wires=[0, i])
        for i in range(1,n_qubits):
            qml.IsingZZ(weights[1][i-1], wires=[0, i])
        for i in range(1,n_qubits):
            qml.IsingYY(weights[2][i-1], wires=[0, i])
        
    @qml.qnode(dev, interface="autograd")
    def PQC(weights, x):
        qml.AngleEmbedding(x,wires=range(n_qubits)[1:])   # Features x are embedded in rotation angles
        for j in range(n_layers):
            block(weights[j])
        return qml.expval(qml.PauliZ(wires=0))
    
    if bool_test == True:
        print(PQC(weights, x))
    if bool_draw == True:
        print(qml.draw(PQC,expansion_strategy ="device")(weights,x))
            
    return PQC