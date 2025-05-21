import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionClassifier:
    def __init__(self, data, epoch):
        self.X = self.normalize(data)
        self.Y = data[:,1]
        self.logit = np.zeros(self.Y.shape)
        
        self.w = 0
        self.b = 0
        self.lr = 0.01
        self.epoch = epoch
        self.samples = self.X.shape[0]
        
        self.feedforward()
        self.predict()
        self.plot()
    
    def normalize(self,data):
        self.X_val = data[:, 0]
        X_mean = np.mean(self.X_val) 
        X_stddev = np.std(self.X_val)
        
        X_normalized = (self.X_val - X_mean) / X_stddev #zscore standardization (from https://web.stanford.edu/~jurafsky/slp3/5.pdf )
        return X_normalized
        
    def feedforward(self):
        for i in range(self.epoch):
            Z = np.dot(self.X,self.w) + self.b
            self.logit = 1 / (1 + np.exp(-Z))
            
            logloss = -(1/self.samples) * np.sum( (self.Y * np.log(self.logit)) + ((1 - self.Y) * np.log(1 - self.logit)) )
            self.w -= self.lr * ( (1/self.samples) * np.dot((self.logit - self.Y), self.X.T) )
            self.b -= self.lr * ( (1/self.samples) * np.sum(self.logit - self.Y))
            
    def predict(self):
        Z = np.dot(self.X,self.w) + self.b
        self.logit = 1 / (1 + np.exp(-Z))
        
        print("Test Accuracy: ", np.mean(self.Y == (self.logit > 0.5).astype(int)) * 100 )
        
    def plot(self):
        X_plot = np.linspace(-self.X.max(), self.X.max(), 2*self.X_val.shape[0])
        Z_plot = np.dot(X_plot,self.w) + self.b
        logit_plot = 1 / (1 + np.exp(-Z_plot))
        decision_boundary = - self.b / self.w
        
        plt.figure(figsize=(12, 10))
        plt.scatter(self.X[self.Y == 0], self.Y[self.Y == 0], color='red', label='Class 0')
        plt.scatter(self.X[self.Y == 1], self.Y[self.Y == 1], color='blue', label='Class 1')
        
        plt.scatter(self.X[self.logit <= 0.5], self.logit[self.logit <= 0.5], color='red', label='Class 0 Predicted')
        plt.scatter(self.X[self.logit > 0.5], self.logit[self.logit > 0.5], color='blue', label='Class 1 Predicted')
        
        plt.plot(X_plot, logit_plot, color='black', linewidth=2, label='Probability Curve (Sigmoid)')
        plt.axhline(y = 0.5, color='green', linestyle='--', label='P(x) = 0.5')
        plt.axvline(x = decision_boundary, color='green', linestyle='--', label='Decision Boundary')
        
        
        plt.xlabel("Normalized Feature Values")
        plt.ylabel("Probability")
        plt.title("Logistic Regression Curve")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 1.1)

        plt.show()
    
        
        
            
data = np.genfromtxt("binary_classification_dataset.csv", delimiter = ",", skip_header = 1)
classify = LogisticRegressionClassifier(data, 5000)





