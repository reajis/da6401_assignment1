import os
import numpy as np
import argparse
from sklearn.metrics import f1_score
from ann.r_neural_network import NeuralNetwork


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SRC_DIR, "best_model.npy")

best_config = argparse.Namespace(
    dataset="mnist",
    epochs=2,
    batch_size=64,
    loss="cross_entropy",
    optimizer="sgd",
    weight_decay=0.0,
    learning_rate=0.01,
    num_layers=2,
    hidden_size=[128, 64],   # replace with your actual best config
    activation="relu",      # replace if needed
    weight_init="xavier"    # replace if needed
)

model = NeuralNetwork(best_config)

weights = np.load(MODEL_PATH, allow_pickle=True).item()
model.set_weights(weights)

X_test = np.random.rand(100, 784)
y_true = np.random.randint(0, 10, size=(100,))

y_pred, _ = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("F1 Score:", f1_score(y_true, y_pred_labels, average="macro"))
print("Model loading test passed.")