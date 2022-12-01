import numpy as np
import math


class mlp():

    def __init__(self, weights, inputs, bias, output, learning_rate):
        self.weights = weights
        self.inputs = inputs
        self.bias = bias
        self.output = output
        self.learning_rate = learning_rate

    def multilayer_perceptron(self, epoches):
        while epoches:
            activation_node_5 = np.dot(self.inputs, self.weights[0]) + self.bias[0]
            activation_node_6 = np.dot(self.inputs, self.weights[1]) + self.bias[1]

            output_node_5 = self._sigmoid(activation_node_5)
            output_node_6 = self._sigmoid(activation_node_6)

            activation_node_7 = self.weights[2][0]*output_node_5 + self.weights[2][1] * output_node_6 + self.bias[2]

            output_node_7 = self._sigmoid(activation_node_7)

            self._update_weights(output_node_5, output_node_6, output_node_7)
            epoches -= 1

    def _update_weights(self, output5, output6, output7):

        error7 = round(output7 * (1 - output7) * (self.output - output7),5)
        error6 = round(output6 * (1 - output6) * error7 * self.bias[1],5)
        error5 = round(output5 * (1 - output5) * error7 * self.bias[0],5)

        self.bias[0] += self.learning_rate * error5
        self.bias[1] += self.learning_rate * error6
        self.bias[2] += self.learning_rate * error7

        for i in range(len(self.weights[0])):
            self.weights[0][i] += self.learning_rate * error5 * self.inputs[i]

        for i in range(len(self.weights[1])):
            self.weights[1][i] += self.learning_rate * error6 * self.inputs[i]

    def predict(self, input):
        
        activation_node_5 = np.dot(input, self.weights[0]) + self.bias[0]
        activation_node_6 = np.dot(input, self.weights[1]) + self.bias[1]

        output_node_5 = self._sigmoid(activation_node_5)
        output_node_6 = self._sigmoid(activation_node_6)

        activation_node_7 = self.weights[2][0]*output_node_5 + self.weights[2][1] * output_node_6 + self.bias[2]

        output_node_7 = self._sigmoid(activation_node_7)
        return round(output_node_7)

    def _sigmoid(self, output):
        return 1/(1+math.exp(-output))


weights = [[0.3, -0.2, 0.2, 0.1], [0.1, 0.4, -0.3, 0.4], [-0.3, 0.2]]
inputs = [1, 1, 0, 1]
bias = [0.2, 0.1, -0.3]
learning_rate = 10
output = 1

perceptron = mlp(weights, inputs, bias, learning_rate, output)
print("Weights before model training : ", weights)
perceptron.multilayer_perceptron(10)
print("\nWeights after model training : ", perceptron.weights)
print(perceptron.predict([1, 1, 0, 1]))
