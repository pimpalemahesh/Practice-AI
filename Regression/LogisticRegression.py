import numpy as np
import math
from statistics import mean

class LogisticRegression():
    
    def __init__(self):
        self.X = []
    
    def fit(self, X, y):
        self.input = X
        self.output = y

        self.a1 = (mean(self.input*self.output) - mean(self.input)*mean(self.output))/(mean(self.input**2) - mean(self.input)**2)
        self.a0 = mean(self.output) - self.a1*mean(self.input)
        
    def predict(self, X, value = 0.5):
        return [self._sigmoid(x, value) for x in X]
    
    def _sigmoid(self, x, value):
        return 1 if 1 / (1 + math.exp(-(self.a0 + self.a1 * x))) >= value else 0

    def r_squared(self, y_output, y_predicted):
        tss = sum((y - mean(y_output))**2 for y in y_output)
        rss = sum((y - x)**2 for y,x in zip(y_output,y_predicted))
        return 1 - (rss/tss)
    
    def mean_square_error(self, y, y_pred):
        return sum([(x-y)**2 for x,y in zip(y, y_pred)])/len(y_pred)
    

input = np.array([0.5, 1.0, 1.25, 2.5, 3.0, 1.75, 4.0, 4.25, 4.75, 5.0])
output = np.array([0,0,0,0,0,1,1,1,1,1])

lr = LogisticRegression()
ot = lr.fit(input, output)
predicted = lr.predict(input, 0.6)
print(predicted)
print(lr.r_squared(output, predicted))
print(lr.mean_square_error(output, predicted))