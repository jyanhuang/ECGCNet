# Import required packages
import random
import time

import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

# assuming X is feature matrix, and Y is corresponding label
# Please replace the following X and Y with your actual data
# X's each row is a signal feature, and each element of y is the corresponding category label
# Appropriate revisions may be conducted according your requirements
import numpy as np

# Setting random seed for the results can be reproducible
from utils import load_data
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)

np.random.seed(42)

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

# X_train,y_train is the training set
# X_test,y_test is the test set
X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])
train_dataset, test_dataset = ECGDataset(X_train, y_train), ECGDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# creating the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=config['seed'])

# training on training set
rf_classifier.fit(X_train, y_train)

start_time = time.time()
# prediction on testing set
y_pred = rf_classifier.predict(X_test)
end_time = time.time()
inference_time=end_time-start_time

# computing accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"test accuracy: {accuracy}")


print(f"aver running time: {inference_time:.4f} 秒")