import torch.nn as nn
import torch

import gzip
import pickle, os
import numpy as np


np.random.seed(0)
torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)

save_base = '../../save'
tag = 'torch_20241018_SGD_lr1e-4'
tag = 'torch_20241019'

_log_path = None
def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


save_folder = os.path.join(save_base, tag)

os.makedirs(save_folder, exist_ok=True)
os.system(f'cp -r ./*.py {save_folder}')
set_log_path(save_folder)

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        # self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                            padding=padding, stride=stride, bias=False)
        # self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        # self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                            padding=padding, bias=False)
        # self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        self.convs = nn.Sequential(
             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
             nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
             nn.ReLU(),
             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding, bias=False),
             nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
             nn.ReLU(),
        )

        # if stride != 1:
        #     # in case stride is not set to 1, we need to downsample the residual so that
        #     # the dimensions are the same when we add them together
        #     self.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        #     )
        # else:
        #     self.downsample = None

        # self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # residual = x

        # out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        # out = self.conv_res2_bn(self.conv_res2(out))

        # # if self.downsample is not None:
        # #     residual = self.downsample(residual)

        # out = self.relu(out)
        out = self.convs(x)+x
        return out


class Net(nn.Module):
    """
    A Residual network.
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Identity(),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=256*7*7, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]
def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


    


def load():
    with open("./mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zero vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float32)

def one_hot_decoding(y_pred):
    return torch.argmax(y_pred, dim=1)

# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值溢出
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    return ( (torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)) *1.0 ).mean()

if __name__ == '__main__':
    save_mnist()
    x_train, y_train, x_test, y_test = load()
    x_train_mean = x_train.mean() 
    x_train_std = x_train.std() 
    # if 1:
    #     x_train = (x_train - x_train_mean) / x_train_std
    #     x_test = (x_test - x_train_mean) / x_train_std # no knowledge of testset
    # x_train = x_train.reshape(x_train.shape[0], 28, 28)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28)
    y_train_one_hot = one_hot_encoding(y_train)
    y_test_one_hot = one_hot_encoding(y_test)


epochs = 200
epoch_test = 1
batch_size = 64
learning_rate = 1e-4
input_size = 784  # MNIST 输入大小 28x28
# hidden_size = 64  # 隐藏层大小
output_size = 10  # 10 个类别

device = 'cuda:2'
x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_train_one_hot = torch.from_numpy(y_train_one_hot).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

y_test_one_hot = torch.from_numpy(y_test_one_hot).to(device)

model = Net().to(device)

# 训练神经网络
from tqdm import tqdm
loss_iter = torch.zeros((epochs, 60000 // batch_size + 1))
loss_epoch = torch.zeros((epochs, 1))
train_accuracy_epoch = torch.zeros((epochs, 1))
test_accuracy_epoch = torch.zeros((epochs, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    # 随机打乱数据
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    model.train()
    
    # for i in range(0, x_train.shape[0], batch_size):
    for i in tqdm(range(0, x_train.shape[0], batch_size), leave=False, desc='train', ncols=0):
        x_batch = x_train[indices[i:i+batch_size]]
        y_batch = y_train_one_hot[indices[i:i+batch_size]]

        optimizer.zero_grad()
        # 前向传播
        y_pred = model(1.0 * x_batch.reshape(x_batch.shape[0], 1,28,28))

        # 计算损失
        loss = loss_fn( y_pred,y_batch)
        loss_iter[epoch, i//batch_size] = loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
    # 每个 epoch 输出一次训练集的损失和准确率
        i = 0
        val_indices = np.arange(x_train.shape[0])
        x_batch = x_train[val_indices[i:i+batch_size]]
        train_pred = model(1.0 * x_batch.reshape(x_batch.shape[0], 1,28,28))
        for i in tqdm(range(batch_size, x_train.shape[0], batch_size), leave=False, desc='train', ncols=0):
            x_batch = x_train[val_indices[i:i+batch_size]]
            train_pred = torch.cat((train_pred, model(1.0*x_batch.reshape(x_batch.shape[0], 1,28,28)) ), dim=0)

        # train_pred = nn.forward(x_train.reshape(x_train.shape[0], 1,28,28))
        train_loss = torch.nn.CrossEntropyLoss()(train_pred, y_train_one_hot )
        train_acc = accuracy(y_train_one_hot, train_pred)
        loss_epoch[epoch] = train_loss.item()
        train_accuracy_epoch[epoch] = train_acc
        
        log(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        if (epoch % epoch_test == 0) or (epoch == epochs - 1):
            i = 0
            test_indices = np.arange(x_test.shape[0])
            x_batch = x_test[test_indices[i:i+batch_size]]
            test_pred = model(1.0 * x_batch.reshape(x_batch.shape[0], 1,28,28))
            for i in tqdm(range(batch_size, x_test.shape[0], batch_size), leave=False, desc='test', ncols=0):
                x_batch = x_test[test_indices[i:i+batch_size]]
                test_pred = torch.cat((test_pred, model(1.0 *x_batch.reshape(x_batch.shape[0], 1,28,28))), dim=0)
            # test_pred = nn.forward(x_test.reshape(x_test.shape[0], 1,28,28))
            test_acc = accuracy(y_test_one_hot, test_pred)
            log(f"Test Accuracy: {test_acc:.4f}")
            test_accuracy_epoch[epoch] = test_acc
            pred_indices = one_hot_decoding(test_pred).type(torch.uint8)
            label_pred_indices = torch.cat((y_test.unsqueeze(1), pred_indices.unsqueeze(1)), dim=1) # 10000, 2
            
            torch.save({'model':model,
                        'loss_iter': loss_iter,
                        'loss_epoch':loss_epoch,
                        'train_accuracy_epoch':train_accuracy_epoch,
                        'test_accuracy_epoch':test_accuracy_epoch,
                        'label_pred_indices':label_pred_indices,
            }, save_folder + os.sep+ f'checkpoint_epoch{epoch}.pt')

            # with open(save_folder + os.sep+ f'checkpoint_epoch{epoch}.pkl', 'wb') as f:
            #     pickle.dump((nn, loss_iter, loss_epoch, train_accuracy_epoch, label_pred_indices), f)