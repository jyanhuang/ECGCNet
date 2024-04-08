import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.3,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
    }
# setting random number seed
seed_value = config['seed']
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# X_train,y_train is the training set, X_test,y_test is the test set
X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])

# Defining ensemble learning parameters
num_classifiers = 5
subspace_size = int(np.sqrt(X_train.shape[1]))

# Creating Subspace Discriminator Set
subspace_classifiers = []

for i in range(num_classifiers):
    # Random select subspace features
    subspace_features = np.random.choice(X_train.shape[1], size=subspace_size, replace=False)

    # Creating basic classifier (decision tree) and training on subspace
    base_classifier = DecisionTreeClassifier(random_state=42)
    base_classifier.fit(X_train[:, subspace_features], y_train)

    #Save subspace features and trained classifiers to a set
    subspace_classifiers.append((subspace_features, base_classifier))


# Ensemble learning prediction
def ensemble_predict(subspace_classifiers, X):
    predictions = np.zeros((X.shape[0], len(subspace_classifiers)))

    for i, (subspace_features, classifier) in enumerate(subspace_classifiers):
        X_subspace = X[:, subspace_features]
        predictions[:, i] = classifier.predict(X_subspace)

    # Taking the voting result as the final prediction
    final_predictions = np.argmax(predictions, axis=1)
    return final_predictions


# Making predictions on the test set
start_time = time.time()
y_pred_ensemble = ensemble_predict(subspace_classifiers, X_test)
end_time = time.time()
inference_time=end_time-start_time
print("average times: {inference_time:.4f} seconds")

# Computing accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print("Accuracy of testing (Ensemble Subspace Discriminant): {accuracy_ensemble}")