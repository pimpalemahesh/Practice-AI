import numpy as np
from statistics import mean
from sklearn.linear_model import LinearRegression

class Simple_Linear_Regression():

    def __init__(self, X, Y):
        self.input = np.array(X)
        self.output = np.array(Y)
        
        self.a1 = (mean(self.input*self.output) - mean(self.input)*mean(self.output))/(mean(self.input**2) - mean(self.input)**2)

        self.a0 = mean(self.output) - self.a1*mean(self.input)

    def calculate(self, input):
        return [self.a1 * x + self.a0 for x in input]
    
    def r_squared(self, y_output, y_predicted):
        tss = sum((y - mean(y_output))**2 for y in y_output)
        rss = sum((y - x)**2 for y,x in zip(y_output,y_predicted))
        return 1 - (rss/tss)
    
    def mean_square_error(self, y, y_pred):
        return sum([(x-y)**2 for x,y in zip(y, y_pred)])/len(y_pred)

# inputs = [6.2, 6.5, 5.4, 6.5, 7.1, 7.9, 8.5, 8.9, 9.5, 10.6]
# outputs = [26.3, 26.6, 25, 26, 27.9, 30.4, 35.4, 38.5, 42.6, 48.3]

inputs = [1, 2, 3, 4, 5]
outputs = [1.2, 1.8, 2.6, 3.2, 3.8]

x = 7
slr = Simple_Linear_Regression(inputs, outputs)
predicted_output = slr.calculate(inputs)
print("Output : ", outputs)
print("Predicted Output : ", predicted_output)
print("R-Squared performance : ", slr.r_squared(outputs, predicted_output))
print()
print("Mean squared Error : " , slr.mean_square_error(outputs, predicted_output))
print()


X = [inputs]
y = [outputs]

lr = LinearRegression()
lr.fit(X, y)
print(lr.predict(X))
print(slr.a0, slr.a1)





        