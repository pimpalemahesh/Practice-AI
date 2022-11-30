import numpy as np
import pandas as pd
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class MLR():

    def __init__(self, X, Y):

        self.input = np.array(X)
        self.output = np.array(Y)

        new_inputs = np.insert(self.input, 0, 1, axis=1)
        
        self.theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(new_inputs),new_inputs)), np.transpose(new_inputs)), self.output)
        
    def predict(self, input):
        return [round(self.theta[0] + self.theta[1]*x[0] + self.theta[2]*x[1],8) for x in input]
        
    def r_squared(self, y_output, y_predicted):
        tss = sum((y - mean(y_output))**2 for y in y_output)
        rss = sum((y - x)**2 for y,x in zip(y_output,y_predicted))
        return 1 - (rss/tss)
        
    def mean_square_error(self, y, y_pred):
        return sum([(x-y)**2 for x,y in zip(y, y_pred)])/len(y_pred)


inputs = [[1,2],[2,3],[3,1],[4,5],[5,4],[6,3],[7,6],[8,4],[9,8]]
outputs = [3,4,6,8,10,11,12,15,16]

mlr = MLR(inputs, outputs)
output_predicted = mlr.predict(inputs)

print("R-squared performance : ", mlr.r_squared(outputs, output_predicted))
print("Mean Square Error : " , mlr.mean_square_error(outputs, output_predicted))

# using sklearn
X = np.array(inputs)
y = np.array(outputs)

lr = LinearRegression()
lr.fit(X, y)
print(r2_score(outputs, output_predicted))

print("Output using my model : " , output_predicted)
print()
print("Output using sklearn model : " , lr.predict(X))
