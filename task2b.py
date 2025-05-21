import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, hidden_no, output_dim, learning_rate, epoch):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_no = hidden_no
        
        self.weights = []
        self.biases = []
        self.initialize_wb()
        self.lr = learning_rate
        self.epoch = epoch
        
        
    def initialize_wb(self):
        self.weights.append(np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2. / self.input_dim)) # He initialized weights
        self.biases.append(np.zeros((1, self.hidden_dim)))
        
        for i in range(self.hidden_no - 1):
            self.weights.append(np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2. / self.hidden_dim)) 
            self.biases.append(np.zeros((1, self.hidden_dim)))
            
        self.weights.append(np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2. / self.hidden_dim))
        self.biases.append(np.zeros((1, self.output_dim)))
        
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, a):
        return (a > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z)) #To prevet overflow, subtracted my maximum value
        return exp_z / exp_z.sum(axis = 1, keepdims=True)
    
    def feedforward(self, X_input):
        activations = [X_input.T.reshape(1,-1)]
        
        for i in range(self.hidden_no):
            activations.append(self.relu( np.dot(activations[-1], self.weights[i]) + self.biases[i] ))

        activations.append(self.softmax(np.dot(activations[-1], self.weights[-1]) + self.biases[-1]))
        
        return activations

    def backprop(self, Y_val, activations):
        deltas = [(activations[-1] - Y_val.reshape(1,-1))] #Softmax + Cross entropy Gradient del = Z - y
        
        for i in range(self.hidden_no, 0, -1):
            
            deltas.append(np.dot(deltas[-1], self.weights[i].T) * self.relu_derivative(activations[i]) )

        deltas.reverse()
            
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= self.lr * np.sum(deltas[i], axis = 0, keepdims=True)
            
    def training(self, X_input, Y_val):
        for i in range(10000): #or X_input.shape[0]
            for j in range(self.epoch):
                self.backprop(Y_val[i], self.feedforward(X_input[i]))
                
            print("Sample:", i + 1,"completed" )
            
    def predict(self, X):
        predictions = []
        
        for i in range(X.shape[0]):
            predictions.append(np.argmax(self.feedforward(X[i])[-1], axis = 1))
            
        output = np.array(predictions)
        return output
    
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28*28) / 255.0 #To flatten and normallize the data
X_test = X_test.reshape(-1, 28*28) / 255.0

Y_train_encoded = to_categorical(y_train, num_classes=10)
Y_test_encoded = to_categorical(y_test, num_classes=10)

model = NeuralNetwork(input_dim = X_train.shape[1], hidden_dim = 64, hidden_no = 2, output_dim = 10, learning_rate = 0.2, epoch = 20)
model.training(X_train, Y_train_encoded)

predictions = model.predict(X_test)
true_labels = y_test

accuracy = accuracy_score(true_labels, predictions)
print("Test accuracy:", accuracy)