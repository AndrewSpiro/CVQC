import pennylane as qml
from DataPreprocessing import r
import jax
import jax.numpy as jnp
from pennylane import numpy as np

def initialize_Ising_circuit(n_qubits: int = r+1, n_layers: int = 2, seed = 0, key = jax.random.PRNGKey(0), bool_test = False, bool_draw = False):
    '''
    Creates a circuit with Ising architecture (Emmanoulopuolos and Dimoska) with a specific number of qubits and layers. Also initializes encodings and weights.
    
            Parameters:
                    n_qubits (int): Number of qubits in the circuit. By default, it is equal to the number of component signals in the toy data.
                    n_layers (int): Number of layers i.e, the number of time the circuit repeats the block. The default is the same as was used by Emmanoulopuolos and Dimoska
                    seed (Any): Seed for randomly initializing the weights and encodings.
                    bool_test (bool): If True, the circuit will be evaluated with the randomly initialized weights and inputs and the output will be printed
                    bool_draw (bool): If True, a drawing of the circuit will be printed
            Returns:
                    PQC (func): The circuit with initialized weights and inputs. It is a function which takes weights of size= (n_layers, 3, n_qubits - 1) and inputs of size = (n_qubits-1).
                    weights: parameters of the PQC to be updated by a optimizer to change output.
                    n_qubits (int): Number of qubits in the circuit.

    '''
    
    dev = qml.device('default.qubit.jax', wires= n_qubits, prng_key = key)
    
    weights = jax.random.uniform(key, minval=0, maxval= jnp.pi, shape=(n_layers,3,n_qubits - 1))
    x = jax.random.uniform(key, minval = 0, maxval = jnp.pi, shape = (n_qubits-1,))
    
    def block(weights):
        for i in range(1,n_qubits):
            qml.IsingXX(weights[0][i-1], wires=[0, i])
        for i in range(1,n_qubits):
            qml.IsingZZ(weights[1][i-1], wires=[0, i])
        for i in range(1,n_qubits):
            qml.IsingYY(weights[2][i-1], wires=[0, i])
    
    @qml.qnode(dev, interface = "jax")
    def circuit(weights, x):
        '''
        Parametrized Quantum Circuit with Ising architecture from Emmanoulopuolos and Dimoska.
                Parameters:
                        weights (numpy.ndarray): Weights to be varied during training
                                Required size: n_layers, 3, n_qubits - 1
                        x (numpy.ndarray): Data to be encoded into the PQC. Should be scaled
                                Required size: n_qubits - 1
                Returns:
                        qml.expval(qml.PauliZ(wire = 0)): Expectation value after applying the Pauli Z operator.
                '''
        qml.AngleEmbedding(x,wires=range(n_qubits)[:-1])   # Features x are embedded in rotation angles
        for j in range(n_layers):
            block(weights[j])
        return qml.expval(qml.PauliZ(wires=n_qubits-1))  
    
    vcircuit = jax.vmap(circuit, in_axes = (None,0), out_axes = 0)  
     
    if bool_test == True:
        print(circuit(weights, x))
    if bool_draw == True:
        print(qml.draw(circuit,expansion_strategy ="device")(weights,x))
            
    return vcircuit, circuit, weights, n_qubits