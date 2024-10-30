# import numpy as np
import cupy as np
from urllib import request
import gzip
import pickle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

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

def init():
    # download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]




def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zero vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float32)

# 定义激活函数和损失函数
def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = -np.log(y_pred[range(n_samples), y_true.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


if __name__ == '__main__':
    init()
    x_train, y_train, x_test, y_test = load()
    # x_train = x_train.reshape(x_train.shape[0], 28, 28)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28)
    y_train_one_hot = one_hot_encoding(y_train)
    y_test_one_hot = one_hot_encoding(y_test)


# 定义神经网络
class resnet9:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # 前向传播
        X = X.reshape(X.shape[0], -1)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backward(self, X, y_true, y_pred, learning_rate):
        # 反向传播
        X = X.reshape(X.shape[0], -1)
        n_samples = X.shape[0]

        # 输出层梯度
        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / n_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

        # 隐藏层梯度
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.A1 > 0)
        dW1 = np.dot(X.T, dZ1) / n_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

        # 更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

epochs = 10
batch_size = 64
learning_rate = 1e-4
input_size = 784  # MNIST 输入大小 28x28
hidden_size = 64  # 隐藏层大小
output_size = 10  # 10 个类别
nn = resnet9(input_size, hidden_size, output_size)
# nn = ResNet9(num_classes=10)
# 训练神经网络
from tqdm import tqdm
for epoch in range(epochs):
    # 随机打乱数据
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    
    # for i in range(0, x_train.shape[0], batch_size):
    for i in tqdm(range(0, x_train.shape[0], batch_size), leave=False, desc='train', ncols=0):
        x_batch = x_train[indices[i:i+batch_size]]
        y_batch = y_train_one_hot[indices[i:i+batch_size]]

        # 前向传播
        y_pred = nn.forward(x_batch.reshape(x_batch.shape[0], 1,28,28))
        # y_pred = nn.forward(x_batch)

        # 计算损失
        loss = cross_entropy_loss(y_batch, y_pred)

        # 反向传播并更新权重
        nn.backward(x_batch.reshape(x_batch.shape[0], 1,28,28), y_batch, y_pred, learning_rate)
    
    # 每个 epoch 输出一次训练集的损失和准确率
    train_pred = nn.forward(x_train)
    train_loss = cross_entropy_loss(y_train_one_hot, train_pred)
    train_acc = accuracy(y_train_one_hot, train_pred)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

# 测试神经网络
test_pred = nn.forward(x_test.reshape(x_test.shape[0], 1,28,28))
# test_pred = nn.forward(x_test)
test_acc = accuracy(y_test_one_hot, test_pred)
print(f"Test Accuracy: {test_acc:.4f}")
