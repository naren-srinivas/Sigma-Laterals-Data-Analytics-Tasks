import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

class Models:
    def __init__(self):
        
        self.models = {
            
            "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"),
            
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1),
            
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.3, use_label_encoder=False, eval_metric="mlogloss"),
            
            "SVM": None,
            
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            
            "Naive Bayes": GaussianNB(),
            
            "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, objective="multiclass", random_state=42),
        
            "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.5),
        
            "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000,early_stopping=True)
        }
        
        self.results = {}
        self.cnn_model = None
        self.history = None
        self.best_svm_params = None
        
    def load_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        self.X_train_flat = X_train.reshape(-1, 28*28) / 255.0
        self.X_test_flat = X_test.reshape(-1, 28*28) / 255.0
        self.y_train = y_train
        self.y_test = y_test
        
        self.X_train_cnn = X_train.reshape(-1, 28, 28, 1) / 255.0
        self.X_test_cnn = X_test.reshape(-1, 28, 28, 1) / 255.0
        self.y_train_encoded = to_categorical(y_train)
        self.y_test_encoded = to_categorical(y_test)
        
    def tune_svm(self, sample_size=10000):
        param_grid = [
            {"kernel": ["linear"], "C": [1, 10, 100]},
            {"kernel": ["poly"], "degree": [2, 3], "C": [1, 10]},
            {"kernel": ["rbf"], "gamma": ["scale", "auto", 0.001, 0.01], "C": [1, 10, 100]}
        ]
        
        svc = SVC(probability=True)
        grid_search = GridSearchCV( svc, param_grid, cv=3, n_jobs=-1, verbose=2 )
        
        print("\nStarting SVM hyperparameter tuning...")
        grid_search.fit(self.X_train_flat[:sample_size], self.y_train[:sample_size])
        
        self.best_svm_params = grid_search.best_params_
        print("Best SVM parameters:", self.best_svm_params)
        print("Best CV accuracy:", grid_search.best_score_)
        
        self.models["SVM"] = grid_search.best_estimator_
            
    def train_traditional_models(self, tune_svm):
        if tune_svm:
            self.tune_svm()
            
        for name, model in self.models.items():
            if name == "SVM" and not tune_svm:
                continue 
            
            print("Training:", name)
            
            model.fit(self.X_train_flat, self.y_train)
            y_pred = model.predict(self.X_test_flat)
            
            acc = accuracy_score(self.y_test, y_pred)
            self.results[name] = acc
            print(name,"Accuracy:", acc)
            print(classification_report(self.y_test, y_pred))
    
    def build_cnn(self):
        self.cnn_model = Sequential([
            Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation="relu"),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax")
        ])
        
        self.cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "Precision", "Recall"])
    
    def train_cnn(self, epochs=10, batch_size=128):
        if not self.cnn_model:
            self.build_cnn()
            
        self.history = self.cnn_model.fit( self.X_train_cnn, self.y_train_encoded, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1 )
        
        _, cnn_acc, cnn_precision, cnn_recall = self.cnn_model.evaluate(self.X_test_cnn, self.y_test_encoded, verbose=0)
        self.results["CNN"] = cnn_acc
        
        print("CNN Test Accuracy:", cnn_acc, "\nCNN Test Precision:", cnn_precision, "\nCNN Test Recall:", cnn_recall)
    
    def compare_models(self):
        print("Model Comparison")
        
        for name, acc in sorted(self.results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20} {acc:.4f}")
    
    def plot_cnn_history(self):
        if not self.history:
            print("CNN not trained yet")
            return
            
        plt.figure(figsize=(10,5))
        plt.plot(self.history.history["accuracy"], label="Train Accuracy")
        plt.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("CNN Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    
    def run_full_pipeline(self):
        self.load_data()
        self.train_traditional_models(tune_svm = True)
        
        self.build_cnn()
        self.train_cnn()
        
        self.compare_models()
        self.plot_cnn_history()
        
        if self.best_svm_params:
            print("Best SVM Parameters Found:")
            for param, value in self.best_svm_params.items():
                print(f"{param:15} {value}")


classifier = Models()
classifier.run_full_pipeline()