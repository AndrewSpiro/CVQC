from pennylane import numpy as np
import pennylane as qml
from numpy import pi
#import tensorflow as tf
#from keras.optimizers import Adam

t = np.linspace(0,10,100)
X = np.sin(t)

dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def circuit(weights,x,y): # y is going to be X(i+1) while x is X(i)
    qml.AngleEmbedding(features = [x], wires = [1])  # creating psi(x)
    qml.RZ(weights[0], wires=[1])
    qml.RY(weights[1], wires=[1])
    qml.RZ(weights[2], wires=[1])  # creating U(theta)psi(x)
    qml.AngleEmbedding([y], wires = [2])
    qml.Hadamard(wires=[0])  # performing the cswap test- i.e. measuring the difference between wire 1 and 2
    qml.CSWAP([0, 1, 2])
    qml.Hadamard(wires=[0])
    return qml.expval(qml.PauliZ(0))  # Returns a value between 0 and 1. If 0, the states are the same.

weights = np.random.uniform(0,pi,3)

def cost(weights, x_batch, y_batch):
    loss = 0
    for j in range(len(x_batch)):
        loss += circuit(weights, x_batch[j],y_batch[j])
    return loss

batch_size = 5
x_batch = np.random.randint(0,len(X),batch_size,)
y_batch = x_batch + 1

#print(cost(weights, [0.3], [0.2]))  # Testing cost function

LR = 0.1
#optimizer = Adam(learning_rate=LR)
opt = qml.AdamOptimizer
epochs = 100
cst = []

for steps in range(epochs):
    weights, c, _ = opt.step_and_cost(cost, weights, x_batch, y_batch)
    cst.append(c)
    if steps % 5 == 0:
        print(cost)


#for i in range(epochs):
#    print(loss)
    #for j in range(len(X)-1):  # Computing an comparing Ux(t) with x(t+1) for all t
        #loss = circuit(X(j), X(j+1), weights)
        #with tf.GradientTape() as tape:
            #loss = loss # add the loss instead of resetting and make a cost function
        #gradients = tape.gradient(loss, weights)
        #optimizer.apply_gradients(gradients, weights)

