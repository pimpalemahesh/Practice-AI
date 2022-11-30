import numpy as np


class SLP():

    def __init__(self, num_inputs=3, num_outputs=1):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.bias = bias

        layers = [num_inputs] + [num_outputs]

        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, input):

        net_inputs = np.dot(input, self.weights)
        net_inputs = [x + self.bias for x in net_inputs]
        for i in range(len(net_inputs)):
            if net_inputs[i] > 0:
                net_inputs[i] = 1
            else:
                net_inputs[i] = 0

        return net_inputs

    def train(self, inputs, weights, targets, learning_rate, bias):
        self.activations = inputs
        self.weights = weights
        self.learning_rate = learning_rate
        self.bias = bias

        errors = [0, 0, 0, 0]
        for i in range(10):
            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output
                errors[j] = error
                if error != 0:
                    self._update_weights(j, error)

            is_error_present = False
            for error in errors:
                if (error != 0):
                    is_error_present = True
            if (is_error_present == False):
                break

        print("Training complete!")
        print("=====")

    def _update_weights(self, index, error):
        self.weights[0][0] += self.learning_rate * \
            error * self.activations[index][0]
        self.weights[0][1] += self.learning_rate * \
            error * self.activations[index][1]
        # print(self.weights[0][0], self.weights[0][1])


def accuracy_score(output_given, output_predicted):
    """	score = (y_true - y_pred) / len(y_true) """
    return round(float(sum(output_predicted == output_given))/float(len(output_given)) * 100, 2)


if __name__ == "__main__":

    # create a Multilayer Perceptron with one hidden layer
    items = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([1, 0, 0, 0])
    weights = [np.array([[0.3], [-0.2]])]
    learning_rate = 0.2
    bias = 0.4
    SLP = SLP(2, 1,)

    # train network
    SLP.train(items, weights, targets, learning_rate, bias)

    input = np.array([1, 0])
    output = SLP.forward_propagate(input)
    print(f"for input [1, 0] output is {output}")
    input = np.array([0, 0])
    output = SLP.forward_propagate(input)
    print(f"for input [0, 0] output is {output}")
    input = np.array([1, 1])
    output = SLP.forward_propagate(input)
    print(f"for input [1, 1] output is {output}")
    input = np.array([0, 1])
    output = SLP.forward_propagate(input)
    print(f"for input [0, 1] output is {output}")
    print()
    print("Accurancy is : ", end="")

    output = SLP.forward_propagate(items)
    print(accuracy_score(targets, list(output)), "%")
