# %%
from Circuits.Ising import initialize_Ising_circuit
from DataPreprocessing import full_signal
from Functions.TrainingFuncs import *

circuit, _,weights, n_qubits = initialize_Ising_circuit(bool_draw=True)

x = np.linspace(0,1)
X = np.stack([x,x,x]).T
    
shape_w = weights.shape
def cost(weights):
    weights_sh = np.reshape(weights,shape_w)

    # print(X.shape)
    return np.sum((circuit(weights_sh,X*3)-x/2+0.25)**2)

print(cost(weights.flatten()))
from scipy.optimize import minimize
res = minimize(cost,x0=weights.flatten())
print(res)

weights_sh_opt = np.reshape(res.x,shape_w)
plt.plot(circuit(weights_sh_opt,X*3))
plt.plot(x/2-0.25)

# np.random.seed(0)
# weights = jnp.array(np.random.rand(weights.shape[0],weights.shape[1],weights.shape[2])*2*np.pi)



# scaled_data, scaler = scale_data(full_signal)
# train, test, train_size, test_size, train_ratio, indices = split_train_test(scaled_data, n_qubits, random = True)
# # %%
# print(weights.shape)
# print(weights)
# print(type(weights))
# np.random.seed(0)
# weights = jnp.array(np.random.rand(weights.shape[0],weights.shape[1],weights.shape[2])*2*np.pi)
# print(weights)

# weights, x_t, target_y_t = train_model(train, test, weights, circuit, n_qubits, max_steps = 1, epochs = 50, bool_plot=True, save_plot='Results/TEST/TEST loss', learning_rate = 0.1)


# # %%
# results_and_params = {
#     "n_qubits" : n_qubits,
#     "indices" : indices,
#     "scaler" : scaler,
#     "weights" : weights,
#     "inputs" : x_t,
#     "targets" : target_y_t
# }

# save_results_params(results_and_params, 'Results/TEST/TEST dict')
# # %%