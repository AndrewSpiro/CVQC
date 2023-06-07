import numpy as np
import pennylane as qml
from numpy import pi
import tensorflow as tf
from keras.optimizers import Adam

t = np.linspace(0,10,1000)
X = np.sin(t)
print(X[50])

dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def circuit(x,y,weights): # y is going to be X(i+1) while x is X(i)
    qml.AngleEmbedding(features = x, wires = [1])  # creating psi(x)
    qml.RZ(weights[0], wires=[1])
    qml.RY(weights[1], wires=[1])
    qml.RZ(weights[2], wires=[1])  # creating U(theta)psi(x)
    qml.AngleEmbedding(y, wires = [2])
    qml.Hadamard(wires=[0])  # performing the cswap test- i.e. measuring the difference between wire 1 and 2
    qml.CSWAP([0, 1, 2])
    qml.Hadamard(wires=[0])
    return qml.expval(qml.PauliZ(0))  # Returns a value between 0 and 1. If 0, the states are the same.

weights = np.random.uniform(0,pi,3)

LR = 0.001
optimizer = Adam(learning_rate=LR)
epochs = 100
swap_accuracy = 20  # how many times the swap test is performed for each prediction
loss = 1

for i in range(epochs):
    print(loss)
    for j in range(len(X)-1):  # Computing an comparing Ux(t) with x(t+1) for all t
        loss = circuit(X(j), X(j+1), weights)
        with tf.GradientTape() as tape:
            loss = loss
        gradients = tape.gradient(loss, weights)
        optimizer.apply_gradients(gradients, weights)

