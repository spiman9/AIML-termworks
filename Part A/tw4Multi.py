import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_ = np.array([[0 , 0 ] , [0 , 1] , [1 , 0] , [1 , 1]])
expected_output = np.array([[0] , [1] , [1] , [0]])


#or gate
#input_ = np.array([[0 , 0 ] , [0 , 1] , [1 , 0] , [1 , 1]])
#expected_output = np.array([[0] , [1] , [1] , [1]])

epochs = 10000
#epochs = 1
lr = 0.5

inputLN , hiddenLN , outputLN = 2 , 2 , 1

hidden_weights = np.random.uniform(size = (inputLN , hiddenLN))
hidden_bias = np.random.uniform(size = (1 , hiddenLN))

output_weights = np.random.uniform(size = (hiddenLN , outputLN))
output_bias = np.random.uniform(size = (1 , outputLN))

print("Initial Hidden weights are   : " , end=" ")
print(*hidden_weights)

print("Initial Hidden bias are   : " , end=" ")
print(*hidden_bias)

print("Initial output weights are   : " , end=" ")
print(*output_weights)

print("Initial output bias are   : " , end=" ")
print(*output_bias)



for _ in range(epochs):
    #forward
    hidden_layer_activation = np.dot(input_ , hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output , output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    #backward
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_output = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    #update the weights
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output , axis = 0 , keepdims = True) * lr
    
    hidden_weights += input_.T.dot(d_hidden_output) * lr
    hidden_bias += np.sum(d_hidden_output , axis =0 , keepdims = True)* lr
    
    
print()
print()

print("Final Hidden weights are   : " , end=" ")
print(*hidden_weights)

print("Final Hidden bias are   : " , end=" ")
print(*hidden_bias)

print("Final output weights are   : " , end=" ")
print(*output_weights)

print("Final output bias are   : " , end=" ")
print(*output_bias)


print("\n\n")
print("Prediction after epochs {} is ".format(epochs) )
print(*predicted_output)








