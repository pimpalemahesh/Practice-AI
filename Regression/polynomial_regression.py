import numpy as np
from statistics import mean
class Polynomial_Regression():

    def __init__(self, input, output, n = 1):
        self.input = input
        self.output = output
        self.n = n
        self.weights = []


    def train(self):
        for i in range(len(self.input)):
            input_matrix = [ [0]*3 for j in range(3)]
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

            self.weights = np.dot(np.linalg.pinv(input_matrix),output_matrix)

    def predict(self, input):
        return [round((self.weights[0] + self.weights[1]*x + self.weights[2]*x**2)*10) for x in input]

    def r_squared(self, y_output, y_predicted):
        tss = sum((y - mean(y_output))**2 for y in y_output)
        rss = sum((y - x)**2 for y,x in zip(y_output,y_predicted))
        return 1 - (rss/tss)
    
    def mean_square_error(self, y, y_pred):
        return sum([(x-y)**2 for x,y in zip(y, y_pred)])/len(y_pred)


input = [1,2,3,4,5,6,7,8,9,10]
output = [1,4,9,15,23,35,46,62,75,97]

pr = Polynomial_Regression(input, output)
pr.train()
predicted = pr.predict(input)
print("Predicted output : ",predicted)
print()
print("R-Squared performance : ", pr.r_squared(output, predicted))
print()
print("Mean squared Error : " , pr.mean_square_error(output, predicted))
print()



    