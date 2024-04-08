import os
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm


from utils import load_data, plot_history_torch, plot_heat_map
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# project root path
project_path = "./"
# define log directory
# must be a subdirectory of the directory specified when starting the web application
# it is recommended to use the date time as the subdirectory name
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model_3.pt"

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


# define the dataset class
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

class Channel_conv(nn.Module):
    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv = nn.Conv1d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.channel_conv = nn.Conv1d(self.dim_untouched, self.dim_untouched, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        conv = x2.shape[-1] // self.n_div
        unconv = x2.shape[-1] - conv
        x3, x4 = torch.split(x2, [conv, unconv], dim=-1)
        x3 = self.channel_conv(x3)
        x2 = torch.cat((x3, x4), -1)
        x = torch.cat((x1, x2), 1)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Model(nn.Module):
    def __init__(self, width_multiplier=0.5, resolution_multiplier=0.5):
        nn.Module.__init__(self)
        num_channels = [1,
                        1,
                        round(8 * width_multiplier * resolution_multiplier),
                        round(16 * width_multiplier * resolution_multiplier)]

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = Channel_conv(8, 4)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.BN = nn.BatchNorm1d(1)
        self.flatten = nn.Flatten()  # 扁平化层
        fc_input_size = num_channels[3] * 38

        # Define the fully connected layers with reduced out_features
        self.fc1 = nn.Sequential(
            nn.Linear(150, 16),  # Reduced out_features to 16
            nn.ReLU()  # 激活函数
        )
        self.fc2 = nn.Linear(16, 5)  # Reduced out_features to 5

    def forward(self, x):
        x = x.view(-1, 1, 300)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.pool1(x)
        x = self.BN(x)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = Model()


# define the training function and validation function
def train_steps(loop, model, criterion, optimizer):
    train_loss = []
    train_acc = []
    model.train()

    for step_index, (X, y) in loop:

        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(train_loss),
            "acc": np.mean(train_acc)}


def test_steps(loop, model, criterion):
    test_loss = []
    test_acc = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()

            test_loss.append(loss)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            test_acc.append(acc)
            loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(test_loss),
            "acc": np.mean(test_acc)}


def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    train_loss_acc = []
    test_loss_ls = []
    test_loss_acc = []
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_loss_acc.append(test_metrix['acc'])

        print(f'Epoch {epoch + 1}: '
              f'train loss: {train_metrix["loss"]}; '
              f'train acc: {train_metrix["acc"]}; ')
        print(f'Epoch {epoch + 1}: '
              f'test loss: {test_metrix["loss"]}; '
              f'test acc: {test_metrix["acc"]}')

        writer.add_scalar('train/loss', train_metrix['loss'], epoch)
        writer.add_scalar('train/accuracy', train_metrix['acc'], epoch)
        writer.add_scalar('validation/loss', test_metrix['loss'], epoch)
        writer.add_scalar('validation/accuracy', test_metrix['acc'], epoch)

    return {'train_loss': train_loss_ls,
            'train_acc': train_loss_acc,
            'test_loss': test_loss_ls,
            'test_acc': test_loss_acc}


def main():
    config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.3,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
    }

    # X_train,y_train is the training set
    # X_test,y_test is the test set
    X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])
    train_dataset, test_dataset = ECGDataset(X_train, y_train), ECGDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # define the model
    width_multiplier = 0.5  # Adjust this value as needed
    resolution_multiplier = 0.5  # Adjust this value as needed
    model = Model(width_multiplier, resolution_multiplier)

    # if os.path.exists(model_path):
    #     # 导入预训练模型，跳过训练过程
    #     print('Import the pre-trained model, skip the training process')
    #     model = Model()
    #     model.load_state_dict(torch.load(model_path))
    #     model.to(device)
    #     model.eval()
    # else:
    # 构建CNN模型
    model = Model()
    model = model.to(device)
    # parameters_to_prune = (
    #     (model.fc1[0], 'weight'),
    # )
    #
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.5,
    # )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # print the model structure
    summary(model, (config['batch_size'], X_train.shape[1]), col_names=["input_size", "kernel_size", "output_size"],
            verbose=2)

    # define the Tensorboard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    # train and evaluate model
    history = train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer)
    writer.close()
    # save the model
    # torch.save(model.state_dict(), model_path)
    # plot the training history
    # plot_history_torch(history)

    # predict the class of test data
    y_pred = []
    model.eval()
    model.to(device)  # 将模型移动到GPU上
    torch.cuda.reset_max_memory_allocated()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print("最大显存消耗（bytes）：", max_memory_allocated)
    print("最大显存消耗（MB）：", max_memory_allocated / 1024 / 1024)
    start_time = time.time()
    for i in range(1):
        with torch.no_grad():
            # 记录开始时间

            for step_index, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                pred_result = torch.argmax(pred, dim=1).cpu().numpy()
                y_pred.extend(pred_result)
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print("最大显存消耗（bytes）：", max_memory_allocated)
    print("最大显存消耗（MB）：", max_memory_allocated / 1024 / 1024)

    end_time = time.time()  # 记录结束时间
    inference_time = (end_time - start_time)/1  # 计算推理时间
    print(f'Inference Time: {inference_time} seconds')

    # 将预测结果转移到CPU上，并将其转换为NumPy数组

    # 绘制混淆矩阵热图
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_heat_map(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    # 在绘制混淆矩阵热图之前，确保y_true和y_pred是NumPy数组或列表
    y_true = y_test  # 不需要使用.cpu()方法，直接使用NumPy数组
    y_pred = np.array(y_pred)  # 将预测标签转换为NumPy数组
    y_pred = y_pred[0: 27658]
    plot_heat_map(y_true, y_pred)

    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_pred)
    accuracy = correct_predictions / total_predictions
    print("Total accuracy: ", accuracy)

    class_accuracy = {}

    for i in range(5):  # Assuming there are 5 classes
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)

        correct_predictions = np.sum(mask_true & mask_pred)
        total_predictions = np.sum(mask_true)

        if total_predictions > 0:
            acc = correct_predictions / total_predictions
            class_accuracy[f"Class {i}"] = acc
        else:
            class_accuracy[f"Class {i}"] = 0.0  # Handle the case where there are no samples for a class

    # Print class-wise accuracy
    for class_label, acc in class_accuracy.items():
        print(f"{class_label} Accuracy: {acc:.4f}")

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Calculate average accuracy
    average_accuracy = np.mean(list(class_accuracy.values()))

    # Print average accuracy
    print(f"Average Accuracy: {average_accuracy:.4f}")


if __name__ == '__main__':
    main()
