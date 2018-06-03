import math
import matplotlib.pyplot as plt
import numpy as np

Weight = 0
Bias = 0
learning_rate = 0.00001
input_X = [1.1, 2, 9, 3, 6, 100, 76, 45, 33, 55 , 78, 91, 11]
output_Y = [31, 20, 13, 45, 44, 81, 12, 67, 12, 32, 101, 120, 131]
total_datapoints = len(output_Y)

def update_weight(w, iterations):
    update = 0
    for i in range(0, iterations):
        predicted = calculate_predicted_value(input_X[i])
        update = update + (predicted - output_Y[i])*input_X[i]
    update = update/iterations
    w = w - learning_rate*update
    return w

def update_bias(b, iterations):
    update = 0
    for i in range(0, iterations):
        predicted = calculate_predicted_value(input_X[i])
        update = update + (predicted - output_Y[i])
    update = update/iterations
    b = b - learning_rate*update
    return b

def calculate_predicted_value(input):
    predicted_Y = input * Weight + Bias
    return predicted_Y

def calculate_loss(iterations):
    loss = 0
    for i in range(0, iterations):
        predicted = calculate_predicted_value(input_X[i])
        loss = loss + (predicted - output_Y[i])**2
    loss = math.sqrt(loss)/iterations
    return loss

def train(weight, bias):
    w = weight
    b = bias
    for i in range(0, total_datapoints):
        calculate_predicted_value(input_X[i])
        w = update_weight(w, i + 1)
        b = update_bias(b, i + 1)
        loss = calculate_loss(i + 1)
        print(loss)
    return w, b

epochs = 100000
plt.scatter(input_X, output_Y)
plt.show()
x = np.linspace(-10, 130, 10000)
for i in range(0, epochs):
    Weight, Bias = train(Weight, Bias)
print(Weight, Bias)
plt.scatter(input_X, output_Y)
plt.plot(x, Weight*x + Bias, linestyle='solid')
plt.show()