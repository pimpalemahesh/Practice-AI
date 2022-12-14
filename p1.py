# graph = {
#     '1' : ['2','3','4'],
#     '2' : ['1','5','6'],
#     '3' : ['1'],
#     '4' : ['1','7','8'],
#     '5' : ['2','9','10'],
#     '6' : ['2'],
#     '7' : ['4', '11', '12'],
#     '8' : ['4'],
#     '9' : ['5'],
#     '10' : ['5'],
#     '11' : ['7'],
#     '12' : ['7']
# }

# BFS
# queue = []
# visited = []
# source = '1'

# def bfs(source):
#     queue.append(source)
#     visited.append(source)

#     while queue:
#         a = queue.pop(0)
#         print(a, end="->")
#         for neighbors in graph[a]:
#             if neighbors not in visited:
#                 queue.append(neighbors)
#                 visited.append(neighbors)

# print("Breadth First Search : ")
# bfs('1')


# DFS
# visited = []
# def dfs(source):
#     if source not in visited:
#         print(source, end='->')
#         visited.append(source)
#         for neightbor in graph[source]:
#             dfs(neightbor)

# print("Depth First Search : " )
# dfs('1')

# <------------------------------------------------------------------------------------------------------>

# Travelling salesman problem
# from itertools import permutations
# import math
# def travelling_salesman(graph, source):
#     remaining_nodes = []
#     for i in range(len(graph)):
#         if (i != source):
#             remaining_nodes.append(i)

#     allpaths = permutations(remaining_nodes)
#     minimum_path_weight = math.inf

#     for path in allpaths:
#         current_path_weight = 0
#         temp_source = source
#         for node in path:
#             current_path_weight += graph[temp_source][node]
#             temp_source = node
#         current_path_weight += graph[temp_source][source]
#         minimum_path_weight = min(minimum_path_weight, current_path_weight)
#     return minimum_path_weight

# graph = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
# s = 0
# print(travelling_salesman(graph, s))


# optimal path between 2 nodes

# graph = [[0,4,4,8,9,0,0],
#          [4,0,5,0,0,0,0],
#          [4,5,0,7,0,4,0],
#          [8,0,7,0,3,5,3],
#          [9,0,0,3,0,0,0],
#          [0,0,4,5,0,0,7],
#          [0,0,0,3,0,7,0]
#         ]

# def optimal_path(graph, source, dest):
#     pass

# path = optimal_path(graph, 0, 7)
# print(path)


# <-------------------------------------------------------------------------------------------------------->

# minmax algorithm
# import math
# def minmax(depth, index, isMax, score, height):
#     if height == depth:
#         return score[index]
#     if isMax:
#         return max(minmax(depth+1, index*2, False, score, height), minmax(depth+1, index*2+1, False, score, height))
#     else:
#         return min(minmax(depth+1, index*2, True, score, height), minmax(depth+1, index*2+1, True, score, height))

# score = [2, 3, 5, 9, 0, 1, 7, 5]
# height = math.log2(len(score))
# print("Optimal value = ", minmax(0, 0, True, score, height))


# alpha-beta pruning
# import math

# def alpha_beta(depth, index, isMax, score, height, alpha, beta):

#     if(depth == height):
#         return score[index]

#     if isMax:
#         curr_max = MAX
#         for i in range(2):
#             val = alpha_beta(depth+1, index*2+i, False, score, height, alpha, beta)
#             curr_max = max(val, curr_max)
#             alpha = max(alpha, curr_max)
#         return curr_max

#     else:
#         curr_min = MIN
#         for i in range(2):
#             val = alpha_beta(depth+1, index*2+i, True, score, height, alpha, beta)
#             curr_min = min(curr_min, val)
#             beta = min(curr_min, beta)
#         return curr_min

# values = [15,16,14,13,12,16,18,11,16,14,18,15,13,16,16,14,13,10,14,15,16,15,17,13,15]
# MAX = -math.inf
# MIN = math.inf
# height = math.log2(len(values))
# print("The optimal value is :", alpha_beta(0, 0, True, values, 4, MIN, MAX))


# <----------------------------------------------------------------------------------------------------------->

# naive bayes classifier

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# class Naive_Bayes():

#     def __init__(self):
#         self.feature = []
#         self.likelihoods = {}
#         self.class_priors = {}
#         self.pred_priors = {}
#         self.train_size = 0
#         self.num_feats = 0

#     def fit(self, X, y):
#         self.feature = list(X.columns)
#         self.X_train = X
#         self.y_train = y
#         self.train_size = X.shape[0]
#         self.num_feats = X.shape[1]

#         for feature in self.feature:
#             self.likelihoods[feature] = {}
#             self.pred_priors[feature] = {}

#             for feature_value in np.unique(self.X_train[feature]):
#                 self.pred_priors[feature].update({feature_value: 0})
#                 for class_value in np.unique(self.y_train):
#                     self.likelihoods[feature].update(
#                         {feature_value+"_"+class_value: 0})
#                     self.class_priors.update({class_value: 0})

#         self._class_probability()
#         self._likelihood_probability()
#         self._prior_probability()

#     def predict(self, X):
#         results = []
#         X = np.array(X)

#         for query in X:
#             prob_outcomes = {}
#             for class_value in np.unique(self.y_train):
#                 prior = self.class_priors[class_value]
#                 likelihood = 1
#                 evidence = 1

#                 for feature, feature_value in zip(self.feature, query):
#                     likelihood *= self.likelihoods[feature][feature_value+ '_'+class_value]
#                     evidence *= self.pred_priors[feature][feature_value]

#                 posterior = likelihood * prior / evidence
#                 prob_outcomes[class_value] = posterior

#             result = max(prob_outcomes, key = lambda x : prob_outcomes[x])
#             results.append(result)
#         return np.array(results)

#     def _likelihood_probability(self):
#         for feature in self.feature:
#             for class_value in np.unique(self.y_train):
#                 class_count = sum(self.y_train == class_value)
#                 feature_likelihood = self.X_train[feature][self.y_train[self.y_train==class_value].index.values.tolist()]
#                 feature_likelihood = feature_likelihood.value_counts().to_dict()
#                 for feature_value, count in feature_likelihood.items():
#                     self.likelihoods[feature][feature_value+"_"+class_value] = count/class_count

#     def _prior_probability(self):
#         for feature in self.feature:
#             feature_value = self.X_train[feature].value_counts().to_dict()
#             for feature_value, count in feature_value.items():
#                 self.pred_priors[feature][feature_value] = count/self.train_size

#     def _class_probability(self):
#         for class_value in np.unique(self.y_train):
#             class_count = sum(self.y_train == class_value)
#             self.class_priors[class_value] = class_count/self.train_size


# df = pd.read_csv("student.csv")
# X, y = df.iloc[:, 1:-1], df.iloc[:, -1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# nb = Naive_Bayes()
# nb.fit(X_train, y_train)

# score = accuracy_score(y_test, nb.predict(X_test))
# print(score)


# Dicision Tree Classifier


# <---------------------------------------------------------------------------------------------------------->

# Multilayer Perceptron

# import numpy as np
# import math


# class mlp():

#     def __init__(self, weights, inputs, bias, output, learning_rate):
#         self.weights = weights
#         self.inputs = inputs
#         self.bias = bias
#         self.output = output
#         self.learning_rate = learning_rate

#     def multilayer_perceptron(self, epoches):
#         while epoches > 0:
#             activation_node_5 = np.dot(self.inputs, self.weights[0]) + self.bias[0]
#             activation_node_6 = np.dot(self.inputs, self.weights[1]) + self.bias[1]

#             output_node_5 = self._sigmoid(activation_node_5)
#             output_node_6 = self._sigmoid(activation_node_6)

#             activation_node_7 = self.weights[2][0]*output_node_5 + self.weights[2][1] * output_node_6 + self.bias[2]

#             output_node_7 = self._sigmoid(activation_node_7)

#             self._update_weights(output_node_5, output_node_6, output_node_7)
#             epoches -= 1

#     def _update_weights(self, output5, output6, output7):

#         error7 = round(output7 * (1 - output7) * (self.output - output7),5)
#         error6 = round(output6 * (1 - output6) * error7 * self.bias[1],5)
#         error5 = round(output5 * (1 - output5) * error7 * self.bias[0],5)

#         self.bias[0] += self.learning_rate * error5
#         self.bias[1] += self.learning_rate * error6
#         self.bias[2] += self.learning_rate * error7

#         for i in range(len(self.weights[0])):
#             self.weights[0][i] += self.learning_rate * error5 * self.inputs[i]

#         for i in range(len(self.weights[1])):
#             self.weights[1][i] += self.learning_rate * error6 * self.inputs[i]

#     def predict(self, input):

#         activation_node_5 = np.dot(input, self.weights[0]) + self.bias[0]
#         activation_node_6 = np.dot(input, self.weights[1]) + self.bias[1]

#         output_node_5 = self._sigmoid(activation_node_5)
#         output_node_6 = self._sigmoid(activation_node_6)

#         activation_node_7 = self.weights[2][0]*output_node_5 + self.weights[2][1] * output_node_6 + self.bias[2]

#         output_node_7 = self._sigmoid(activation_node_7)
#         return round(output_node_7)

#     def _sigmoid(self, output):
#         return 1/(1+math.exp(-output))


# weights = [[0.3, -0.2, 0.2, 0.1], [0.1, 0.4, -0.3, 0.4], [-0.3, 0.2]]
# inputs = [1, 1, 0, 1]
# bias = [0.2, 0.1, -0.3]
# learning_rate = 10
# output = 1

# perceptron = mlp(weights, inputs, bias, learning_rate, output)
# print("Weights before model training : ", weights)
# perceptron.multilayer_perceptron(10)
# print("\nWeights after model training : ", perceptron.weights)
# print(perceptron.predict([1, 1, 0, 1]))


# <----------------------------------------------------------------------------------------------------------->

# Simple Regression
# import numpy as np
# from statistics import mean
# from sklearn.model_selection import train_test_split

# class simple_regression():

#     def __init__(self):
#         self.a1 = 0
#         self.a2 = 0

#     def fit(self, X, y):
#         self.input = np.array(X)
#         self.output = np.array(y)
#         self.a1 = (mean(self.input*self.output) - mean(self.input)*mean(self.output)) / (mean(self.input**2) - mean(self.input)**2)
#         self.a0 = mean(self.output) - self.a1*mean(self.input)

#     def predict(self, X):
#         return [self.a1*x + self.a0 for x in X]

#     def r_square(self, y_output, y_predict):
#         RSS = sum((y - mean(y_predict))**2 for y in y_output)
#         TSS = sum((y1 - y2)**2 for y1,y2 in zip(y_output, y_predict))
#         return 1 - RSS/TSS

#     def mean_square_error(self, y_output, y_predicted):
#         return sum((y1-y2)**2 for y1, y2 in zip(y_output, y_predicted))/len(y_output)

# X = [1, 2, 3, 4, 5]
# y = [1.2, 1.8, 2.6, 3.2, 3.8]

# sr = simple_regression()
# sr.fit(X,y)

# predict = sr.predict(X)
# print(predict)

# print("Mean square Error ", sr.mean_square_error(y, predict))


# Logistic Regression
# import numpy as np
# from statistics import mean
# import math
# class Logistic_Regression():

#     def __init__(self):
#         self.a0 = 0
#         self.a1 = 0

#     def fit(self, X, y):
#         self.input = np.array(X)
#         self.output = np.array(y)

#         self.a1 = (mean(self.input*self.output) - mean(self.input)*mean(self.output)) / (mean(self.input**2) - mean(self.input)**2)
#         self.a0 = mean(self.output) - self.a1*mean(self.input)

#     def predict(self, X, value = 0.5):
#         return [self._signmoid(x, value) for x in X]

#     def r_squared(self, y_output, y_predicted):
#         TSS = sum((y - mean(y_output))**2 for y in y_output)
#         RSS = sum((y1-y2)**2 for y1,y2 in zip(y_output, y_predicted))
#         return 1 - RSS/TSS

#     def mean_squared_error(self, y_output, y_predicted):
#         return sum((y1-y2)**2 for y1, y2 in zip(y_output, y_predicted))/len(y_output)

#     def _signmoid(self, x, value):
#         return 1 if (1 / (1 + math.exp(-(x*self.a1 + self.a0)))) >= value else 0

# input = np.array([0.5, 1.0, 1.25, 2.5, 3.0, 1.75, 4.0, 4.25, 4.75, 5.0])
# output = np.array([0,0,0,0,0,1,1,1,1,1])

# lr = Logistic_Regression()
# ot = lr.fit(input, output)
# predicted = lr.predict(input, 0.6)
# print(predicted)
# print(lr.r_squared(output, predicted))
# print(lr.mean_squared_error(output, predicted))


# Multiple Linear Regression
# import numpy as np

# class multiple_linear_regression():

#     def __init__(self):
#         self.theta = []

#     def fit(self, X, y):
#         self.input = np.array(X)
#         self.output = np.array(y)

#         new_matrix = np.insert(X, 0, 1, axis=1)
#         self.theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(new_matrix), new_matrix)), np.transpose(new_matrix)), y)
#         print(self.theta)

#     def predict(self, X):
#         return [round(self.theta[0] + self.theta[1]*x[0] + self.theta[2]*x[1]) for x in X]

# inputs = [[1,2],[2,3],[3,1],[4,5],[5,4],[6,3],[7,6],[8,4],[9,8]]
# outputs = [3,4,6,8,10,11,12,15,16]

# mlr = multiple_linear_regression()
# mlr.fit(inputs, outputs)
# output_predicted = mlr.predict(inputs)
# print(output_predicted)


# Polynomial Regression
import numpy as np
from sklearn.model_selection import train_test_split
import math

class polynomial_regression():

    def __init__(self):
        self.input = []
        self.output = []
        self.weight = [math.inf,math.inf,math.inf]
        self.n = 1

    def fit(self, X, y, n=1):
        self.input = X
        self.output = y
        self.n = n

        for i in range(len(self.input)):
            input_matrix = [[0]*3 for j in range(3)]
            input_matrix[0][0] = self.n
            input_matrix[0][1] = self.input[i]
            input_matrix[0][2] = self.input[i]**2
            input_matrix[1][0] = self.input[i]
            input_matrix[1][1] = self.input[i]**2
            input_matrix[1][2] = self.input[i]**3
            input_matrix[2][0] = self.input[i]**2
            input_matrix[2][1] = self.input[i]**3
            input_matrix[2][2] = self.input[i]**4

            output_matrix = [0, 0, 0]
            output_matrix[0] = self.output[i]
            output_matrix[1] = self.input[i]**2
            output_matrix[2] = self.input[i]**3

            weight = np.dot(np.linalg.pinv(input_matrix), output_matrix)
            if(self.weight[0] > weight[0]):
                self.weight = weight

    def predict(self, X):
        return [round((self.weight[0] + self.weight[1]*x + self.weight[2]*x**2)*10) for x in X]


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 4, 9, 15, 23, 35, 46, 62, 75, 97]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
pr = polynomial_regression()
pr.fit(X_train, y_train)
predicted = pr.predict(X_test)
print("Output : " , y_test)
print("Predicted output : ", predicted)
