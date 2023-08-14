import pennylane as qml
from pennylane import numpy as np
from DataPreprocessing import r
import jax
def initialize_MPS_circuit(n_qubits: int = r+1, n_layers: int = 2, seed = 0, jax_key = 0,bool_test = False, bool_draw = False):

    np.random.seed(seed)
    key = jax.random.PRNGKey(jax_key)
    dev = qml.device('default.qubit.jax', wires= n_qubits, prng_key = key)
    weights = 2 * np.pi * np.random.random(size=(n_qubits-1,n_layers))
    x = 2 * np.pi *np.random.random(size = (n_qubits))
        
    def block(weights, wires):
        qml.RY(weights[0], wires=wires[0])
        qml.RX(weights[1], wires=wires[0])
        qml.RZ(weights[2], wires=wires[1])
        qml.RY(weights[3], wires=wires[1])
        qml.CNOT(wires=wires)
        
    @qml.qnode(dev, interface="jax")
    def MPS_circuit(weights, x):
        '''
        Circuit with Matrix Product State Architecure.

                Parameters: 
                    weights: Should be of shape (n_layers,n_qubits-1)
                    x:
                
                
                '''
        qml.AngleEmbedding(x,wires=range(n_qubits))   # Features x are embedded in rotation angles
        qml.MPS(wires=range(n_qubits), n_block_wires=2,block=block, n_params_block=n_layers, template_weights=weights) # Variational layer
        return qml.expval(qml.PauliZ(n_qubits-1)) # Expectation value of the \sigma_z operator on the last qubit

    
    vcircuit = jax.vmap(MPS_circuit, in_axes = (None,0), out_axes = 0)  

    if bool_test == True:
        print(MPS_circuit(weights, x))
    if bool_draw == True:
        print(qml.draw(MPS_circuit,expansion_strategy ="device")(weights,x))
            
    return vcircuit, MPS_circuit, weights, n_qubits