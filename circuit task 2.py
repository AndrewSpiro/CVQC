import numpy as np
import pennylane as qml
from numpy import pi
import tensorflow as tf
from keras.optimizers import Adam

t = np.linspace(0,10,1000)
X = np.sin(t)

dev = qml.device('default.qubit', wires=3)

def circuit(x,y,weights): # y is going to be X(i+1) while x is X(i)
    qml.AngleEmbedding(features = x, wires = [1])  # creating psi(x)
    qml.RZ(weights[0], wires=[1])
    qml.RY(weights[1], wires=[1])
    qml.RZ(weights[2], wires=[1])  # creating U(theta)psi(x)
    qml.AngleEmbedding(y, wires = [2])
    qml.Hadamard(wires=[0])  # performing the cswap test- i.e. measuring the difference between wire 1 and 2
    qml.CSWAP([0, 1, 2])
    qml.Hadamard(wires=[0])
    return qml.expval(qml.PauliZ)  # This is going to return either a 1 (states are different) or a 0 (states are the same)

weights = np.random.uniform(0,pi,3)

LR = 0.001
optimizer = Adam(learning_rate=LR)
epochs = 100
swap_accuracy = 20  # how many times the swap test is performed for each prediction
loss = 1

for i in range(epochs):
    print(loss)
    for j in range(len(X)-1):  # Computing an comparing Ux(t) with x(t+1) for all t
        M = []
        for k in range(swap_accuracy):
            print("X[j] is" + str(X[j]) + "and X[j + 1] is " + str(X[j+1]))
            M.append ( circuit( [X[j]], [X[j + 1]], weights) )
        s = 1 - 2/len(swap_accuracy) * sum(M)  # M is 0 when they are the same, so s will equal 1.
        with tf.GradientTape() as tape:
            loss = 1 - abs(s) # Loss is 0 when states are the same
        gradients = tape.gradient(loss, weights)
        optimizer.apply_gradients(gradients, weights)

