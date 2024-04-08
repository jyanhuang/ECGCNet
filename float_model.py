import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import load_data

max_value = 0
min_value = 0

def findextreme(tmp): # 这个函数的功能是找到一系列数字钟最大值和最小值。函数接受一个参数'tmp'，表示输入的一个数字。
    global max_value
    global min_value
    if tmp>max_value:
        max_value = tmp
    if tmp<min_value:
        min_value = tmp


def Linear(input_data, inchannel, outnumber, weights, bias):
    output = np.zeros(outnumber)
    input_len = int(len(input_data) / inchannel)
    for k in range(outnumber):
        for i in range(inchannel):
            for j in range(input_len):
                output[k] += input_data[i * input_len + j] * weights[k * len(input_data) + i * input_len + j]
        output[k] += bias[k]
    return output.tolist()

def dropout(X, drop_probability):
    X = np.array(X)
    keep_probability = 1 - drop_probability
    mask = np.random.uniform(0, 1, X.shape) < keep_probability
    if keep_probability > 0.0:
        scale = (1 / keep_probability)
    else:
        scale = 0.0
    return (X * mask * scale).tolist()

def npConv_one_filter1D(img, shape, weight_one_filter, kernel_size=3):
    # 获取输入的维度信息
    length = shape[0]
    result = [0] * length

    for i in range(length):
        output_value = 0
        for m in range(kernel_size):
            weight_index = m
            input_index = i + m
            output_value += weight_one_filter[weight_index] * img[input_index]
            findextreme(output_value)
        result[i] = output_value
        findextreme(result[i])
    return result


def npConv1D(feature, shape, weight, bias, in_channels=1, out_channels=1, kernel_size=3, padding=0, group=None):
    length = shape[0]
    result = np.zeros(out_channels * length)
    if padding != 0:
        pad = [0] * (length + 2 * padding) * in_channels
        for i in range(in_channels):
            pad[i * (length + 2 * padding) + padding:i * (length + 2 * padding) + length + padding] = feature[i*length:(i+1)*length]
        feature = pad
    if group is None:
        for i in range(out_channels):
            for j in range(in_channels):
                weight_one_filter = weight[i * in_channels * kernel_size + j * kernel_size:i * in_channels * kernel_size + (j + 1) * kernel_size]
                result[i * length:(i + 1) * length] += np.array(npConv_one_filter1D(feature[j * (length + 2 * padding):(j + 1) * (length + 2 * padding)], shape, weight_one_filter, kernel_size=kernel_size))
            result[i * length:(i + 1) * length] = result[i * length:(i + 1) * length] + bias[i]
    else:
        for i in range(in_channels):
            weight_one_filter = weight[i * kernel_size:(i + 1) * kernel_size]
            result[i * length:(i + 1) * length] = npConv_one_filter1D(feature[i * (length + 2 * padding):(i + 1) * (length + 2 * padding)], shape, weight_one_filter, kernel_size=kernel_size)
            result[i * length:(i + 1) * length] = result[i * length:(i + 1) * length] + bias[i]
    return result.tolist()

def npMaxPool1D(img, kernel_size, stride, padding=0, inchannel=1):
    length = int(len(img) / inchannel)

    # Calculate the output length
    out_length = int((length + 2 * padding - kernel_size) / stride + 1)

    # Initialize the output result
    result = [0] * out_length * inchannel
    for i in range(inchannel):
        for j in range(out_length):
            # Calculate the pool window position in the input sequence
            start = j * stride - padding
            end = start + kernel_size

            # Get the input sequence window
            window = img[max(0, start):min(length, end)]

            # Choose the maximum value in the window as the output
            result[out_length * i + j] = np.max(window)

    return result


def npAvgPool1D(img, kernel_size, stride, padding=0, inchannel=1):
    length = int(len(img) / inchannel)
    # Calculate the output length
    out_length = int((length + 2 * padding - kernel_size) / stride + 1)

    result = [0] * (out_length * inchannel)  # 使用浮点数存储结果

    # 对每个通道进行填充和平均池化
    for i in range(inchannel):
        pad = [0] * (length + 2 * padding)
        pad[padding:padding + length] = img[i * length:(i + 1) * length]

        for j in range(0, out_length):
            window = pad[j * stride:j * stride + kernel_size]
            result[out_length * i + j] = sum(window) / kernel_size

    return result


def np_nn(img, conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias, conv4_weight, conv4_bias, conv5_weight, conv5_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias): # 是一个神经网络的前向传播函数，用于对输入图像img进行处理，并返回最终的预测结果。
    conv1 = npConv1D(img, [300], conv1_weight, conv1_bias, 1, 1, 15, 7)
    conv1 = npMaxPool1D(conv1, 3, 2, 1)
    conv2 = npConv1D(conv1, [150], conv2_weight, conv2_bias, 1, 1, 17, 8)
    conv2 = npMaxPool1D(conv2, 3, 2, 1)
    conv3 = npConv1D(conv2, [75], conv3_weight, conv3_bias, 1, 2, 19, 9)
    conv3 = npAvgPool1D(conv3, 3, 2, 1, 2)
    conv4 = npConv1D(conv3, [38], conv4_weight, conv4_bias, 2, 2, 21, 10,group=0)
    conv5 = npConv1D(conv4, [38], conv5_weight, conv5_bias, 2, 4, 1)
    fc1 = Linear(conv5, 4, 16, fc1_weight, fc1_bias)
    fc1 = [x if x >= 0 else 0 for x in fc1]
    fc2 = Linear(fc1, 1, 5, fc2_weight, fc2_bias)
    # conv1 = npConv(conv1, (28, 28), weight_conv2, 1, 16, 1, 0)
    # conv1 = [x if x >=0 else 0 for x in conv1]
    # x = npMaxPool(conv1, (16, 28, 28), 3, 2, 1)
    #
    # conv2 = npConv(x, (14, 14), weight_conv3, 16, 32, 3)
    # conv2 = [x if x >= 0 else 0 for x in conv2]
    # x = npMaxPool(conv2, (32, 14, 14), 8, 8)
    #
    # x = dropout(x , 0.1)
    # conv3 = npConv(x, (1, 1), weight_conv4, 32, 10, 3)
    # conv3 = [x if x >= 0 else x*0.02 for x in conv3]
    return fc2




from tqdm import tqdm

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
                        round(4 * width_multiplier * resolution_multiplier),
                        round(8 * width_multiplier * resolution_multiplier),
                        round(16 * width_multiplier * resolution_multiplier)]


        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_channels[0], kernel_size=15, stride=1, padding=7)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels[0], out_channels=num_channels[1], kernel_size=17, stride=1, padding=8)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_channels[1], out_channels=num_channels[2], kernel_size=19, stride=1, padding=9)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = DepthwiseSeparableConv1d(in_channels=num_channels[2], out_channels=num_channels[3], kernel_size=21, stride=1, padding=10)
        self.flatten = nn.Flatten()   # 扁平化层
        fc_input_size = num_channels[3] * 38

        # Define the fully connected layers with reduced out_features
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_size, 16),  # Reduced out_features to 16
            nn.ReLU()   # 激活函数
        )
        self.fc2 = nn.Linear(16, 5)  # Reduced out_features to 5

    def forward(self, x):
        x = x.view(-1, 1, 300)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model_path = "./ecg_model_3.pt"

config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.3,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
    }
X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])

 # Adjust this value as needed
model = Model()
model.load_state_dict(torch.load(model_path))
numpy_weights = {}
name_nn = []
for name, param in model.state_dict().items():
    name_nn.append(name)
    numpy_weights[name] = param.cpu().numpy()



conv1_weight = numpy_weights[name_nn[0]]
conv1_bias = numpy_weights[name_nn[1]]
conv2_weight = numpy_weights[name_nn[2]]
conv2_bias = numpy_weights[name_nn[3]]
conv3_weight = numpy_weights[name_nn[4]]
conv3_bias = numpy_weights[name_nn[5]]
conv4_weight = numpy_weights[name_nn[6]]
conv4_bias = numpy_weights[name_nn[7]]
conv5_weight = numpy_weights[name_nn[8]]
conv5_bias = numpy_weights[name_nn[9]]
fc1_weight = numpy_weights[name_nn[10]]
fc1_bias = numpy_weights[name_nn[11]]
fc2_weight = numpy_weights[name_nn[12]]
fc2_bias = numpy_weights[name_nn[13]]


X_test_np = X_test
Y_test_np = y_test

X_test_np = np.round(X_test_np * (2 ** 8))
conv1_weight = np.round(conv1_weight * (2 ** 8))
conv1_bias = np.round(conv1_bias * (2 ** 8))
conv2_weight = np.round(conv2_weight * (2 ** 8))
conv2_bias = np.round(conv2_bias * (2 ** 8))
conv3_weight = np.round(conv3_weight * (2 ** 8))
conv3_bias = np.round(conv3_bias * (2 ** 8))
conv4_weight = np.round(conv4_weight * (2 ** 8))
conv4_bias = np.round(conv4_bias * (2 ** 8))
conv5_weight = np.round(conv5_weight * (2 ** 8))
conv5_bias = np.round(conv5_bias * (2 ** 8))
fc1_weight = np.round(fc1_weight * (2 ** 8))
fc1_bias = np.round(fc1_bias * (2 ** 8))
fc2_weight = np.round(fc2_weight * (2 ** 8))
fc2_bias = np.round(fc2_bias * (2 ** 8))

X_test_np = X_test_np.flatten()
conv1_weight = conv1_weight.flatten()
conv1_bias = conv1_bias.flatten()
conv2_weight = conv2_weight.flatten()
conv2_bias = conv2_bias.flatten()
conv3_weight = conv3_weight.flatten()
conv3_bias = conv3_bias.flatten()
conv4_weight = conv4_weight.flatten()
conv4_bias = conv4_bias.flatten()
conv5_weight = conv5_weight.flatten()
conv5_bias = conv5_bias.flatten()
fc1_weight = fc1_weight.flatten()
fc1_bias = fc1_bias.flatten()
fc2_weight = fc2_weight.flatten()
fc2_bias = fc2_bias.flatten()

X_test_np = [int(x) for x in X_test_np.tolist()]
conv1_weight = [int(x) for x in conv1_weight.tolist()]
conv1_bias = [int(x) for x in conv1_bias.tolist()]
conv2_weight = [int(x) for x in conv2_weight.tolist()]
conv2_bias = [int(x) for x in conv2_bias.tolist()]
conv3_weight = [int(x) for x in conv3_weight.tolist()]
conv3_bias = [int(x) for x in conv3_bias.tolist()]
conv4_weight = [int(x) for x in conv4_weight.tolist()]
conv4_bias = [int(x) for x in conv4_bias.tolist()]
conv5_weight = [int(x) for x in conv5_weight.tolist()]
conv5_bias = [int(x) for x in conv5_bias.tolist()]
fc1_weight = [int(x) for x in fc1_weight.tolist()]
fc1_bias = [int(x) for x in fc1_bias.tolist()]
fc2_weight = [int(x) for x in fc2_weight.tolist()]
fc2_bias = [int(x) for x in fc2_bias.tolist()]




X_test_np = [float(x/(2**8)) for x in X_test_np]
conv1_weight = [float(x/(2**8)) for x in conv1_weight]
conv1_bias = [float(x/(2**8)) for x in conv1_bias]
conv2_weight = [float(x/(2**8)) for x in conv2_weight]
conv2_bias = [float(x/(2**8)) for x in conv2_bias]
conv3_weight = [float(x/(2**8)) for x in conv3_weight]
conv3_bias = [float(x/(2**8)) for x in conv3_bias]
conv4_weight = [float(x/(2**8)) for x in conv4_weight]
conv4_bias = [float(x/(2**8)) for x in conv4_bias]
conv5_weight = [float(x/(2**8)) for x in conv5_weight]
conv5_bias = [float(x/(2**8)) for x in conv5_bias]
fc1_weight = [float(x/(2**8)) for x in fc1_weight]
fc1_bias = [float(x/(2**8)) for x in fc1_bias]
fc2_weight = [float(x/(2**8)) for x in fc2_weight]
fc2_bias = [float(x/(2**8)) for x in fc2_bias]


# 定义文件名
X_test_np_file = "./float_weight/X_test.txt"
Y_test_file = "./float_weight/Y_test.txt"
conv1_weight_file = "./float_weight/conv1_weight.txt"
conv1_bias_file = "./float_weight/conv1_bias.txt"
conv2_weight_file = "./float_weight/conv2_weight.txt"
conv2_bias_file = "./float_weight/conv2_bias.txt"
conv3_weight_file = "./float_weight/conv3_weight.txt"
conv3_bias_file = "./float_weight/conv3_bias.txt"
conv4_weight_file = "./float_weight/conv4_weight.txt"
conv4_bias_file = "./float_weight/conv4_bias.txt"
conv5_weight_file = "./float_weight/conv5_weight.txt"
conv5_bias_file = "./float_weight/conv5_bias.txt"
fc1_weight_file = "./float_weight/fc1_weight.txt"
fc1_bias_file = "./float_weight/fc1_bias.txt"
fc2_weight_file = "./float_weight/fc2_weight.txt"
fc2_bias_file = "./float_weight/fc2_bias.txt"

# 保存到文件
np.savetxt(X_test_np_file, X_test_np, fmt='%f', delimiter=',')
np.savetxt(Y_test_file, Y_test_np, fmt='%f', delimiter=',')
np.savetxt(conv1_weight_file, conv1_weight, fmt='%f', delimiter=',')
np.savetxt(conv1_bias_file, conv1_bias, fmt='%f', delimiter=',')
np.savetxt(conv2_weight_file, conv2_weight, fmt='%f', delimiter=',')
np.savetxt(conv2_bias_file, conv2_bias, fmt='%f', delimiter=',')
np.savetxt(conv3_weight_file, conv3_weight, fmt='%f', delimiter=',')
np.savetxt(conv3_bias_file, conv3_bias, fmt='%f', delimiter=',')
np.savetxt(conv4_weight_file, conv4_weight, fmt='%f', delimiter=',')
np.savetxt(conv4_bias_file, conv4_bias, fmt='%f', delimiter=',')
np.savetxt(conv5_weight_file, conv5_weight, fmt='%f', delimiter=',')
np.savetxt(conv5_bias_file, conv5_bias, fmt='%f', delimiter=',')
np.savetxt(fc1_weight_file, fc1_weight, fmt='%f', delimiter=',')
np.savetxt(fc1_bias_file, fc1_bias, fmt='%f', delimiter=',')
np.savetxt(fc2_weight_file, fc2_weight, fmt='%f', delimiter=',')
np.savetxt(fc2_bias_file, fc2_bias, fmt='%f', delimiter=',')

correct_num = 0
total_num = 100

for i in tqdm(range(total_num)):
    my_list = np_nn(X_test_np[i * 300:(i + 1) * 300], conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias, conv4_weight, conv4_bias, conv5_weight, conv5_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
    max_value = max(my_list)
    max_index = my_list.index(max_value)
    if Y_test_np[i] == max_index:
        correct_num = correct_num + 1
print("测试集准确率：", correct_num/ (total_num / 100), '%')