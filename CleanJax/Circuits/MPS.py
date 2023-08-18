import pennylane as qml
from pennylane import numpy as np
from DataPreprocessing import r
import jax
def initialize_MPS_circuit(n_qubits: int = r+1, seed = 0, jax_key = 0,bool_test = False, bool_draw = False):
    ''''''

    n_params_b = 4

    np.random.seed(seed)
    key = jax.random.PRNGKey(jax_key)
    dev = qml.device('default.qubit.jax', wires= n_qubits, prng_key = key)
    weights = 2 * np.pi * np.random.random(size=(n_qubits-1,n_params_b))
    x = 2 * np.pi *np.random.random(size = (n_qubits-1))
        
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
                
                
                '''
        qml.AngleEmbedding(x,wires=range(n_qubits)[:-1])   # Features x are embedded in rotation angles
        qml.MPS(wires=range(n_qubits), n_block_wires=2,block=block, n_params_block=n_params_b, template_weights=weights) # Variational layer
        return qml.expval(qml.PauliZ(wires = n_qubits-1)) # Expectation value of the \sigma_z operator on the last qubit

    
    vcircuit = jax.vmap(MPS_circuit, in_axes = (None,0), out_axes = 0)  

    if bool_test == True:
        print(MPS_circuit(weights, x))
    if bool_draw == True:
        qml.drawer.use_style('black_white')
        fig, ax = qml.draw_mpl(MPS_circuit)(weights,x)
        fig.show
            
    return vcircuit, MPS_circuit, weights, n_qubits