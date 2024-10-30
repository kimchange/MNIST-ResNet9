# import numpy as np
import cupy as np
from urllib import request
import gzip
import pickle, os

np.random.seed(0)

save_base = '../save'
tag = '20241019_0.9_lr1e-4'
tag = '20241019'
tag = '20241020'

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


class AvgPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=1):
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = kernel_size[-2]
        self.w_width = kernel_size[-1]

    def __call__(self, x):
        self.x = x
        self.in_height = x.shape[-2]
        self.in_width = x.shape[-1]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        out = np.zeros((x.shape[0], x.shape[1],self.out_height, self.out_width))

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[:,:,i:i+1, j:j+1] = np.mean(x[:,:,start_i: end_i, start_j: end_j], axis=(-2,-1) ,keepdims=True)
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                dx[:,:, start_i: end_i, start_j: end_j] = d_loss[:,:,i:i+1, j:j+1] / (self.w_width * self.w_height)
        return dx


class MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=1):
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = kernel_size[-2]
        self.w_width = kernel_size[-1]

        self.x = None
        self.in_height = None
        self.in_width = None

        self.out_height = None
        self.out_width = None

        self.arg_max = None

    def __call__(self, x):
        self.x = x
        self.in_height = x.shape[-2]
        self.in_width = x.shape[-1]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        out = np.zeros((x.shape[0], x.shape[1],self.out_height, self.out_width))

        self.arg_max_mask = np.zeros_like(x, dtype=np.float32)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[:,:,i:i+1, j:j+1] = np.max(x[:,:,start_i: end_i, start_j: end_j].reshape(x.shape[0],x.shape[1],-1), axis=(-1) , keepdims=False)[:,:,np.newaxis, np.newaxis]
                # self.arg_max[:,:,i:i+1, j] = np.argmax(x[:,:,start_i: end_i, start_j: end_j].reshape(x.shape[0],x.shape[1],-1), axis=(-1) ,keepdims=True)
                # arg_max = np.argmax(x[:,:,start_i: end_i, start_j: end_j].reshape(x.shape[0],x.shape[1],-1), axis=(-1) ,keepdims=True)
        # self.arg_max = self.arg_max
        self.arg_max_mask = ( out.repeat(self.w_height, axis=-2).repeat(self.w_width, axis=-1) == self.x)
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        dx = d_loss.repeat(self.w_height, axis=-2).repeat(self.w_width, axis=-1) * self.arg_max_mask

        # for i in range(self.out_height):
        #     for j in range(self.out_width):
        #         start_i = i * self.stride
        #         start_j = j * self.stride
        #         end_i = start_i + self.w_height
        #         end_j = start_j + self.w_width
        #         # np.put_along_axis( dx[:,:,start_i:end_i, start_j:end_j].reshape(dx.shape[0],dx.shape[1],-1), self.arg_max, d_loss[:,:,i:i+1, j], axis=-1)
        #         # this is a vectorized operation
        #         # seems numpy have it but cupy not
        #         for bb in range(dx.shape[0]):
        #             for cc in range(dx.shape[1]):
        #                 dx[bb,cc,start_i:end_i, start_j:end_j].reshape(-1)[self.arg_max[bb,cc,i, j]] = d_loss[bb,cc,i, j]

        return dx

class Dropout(object):
    def __init__(self, dropout_probability):
        self.dropout_probability = dropout_probability

    def forward(self, inputs):
        self.inputs = inputs
        # https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch
        # to drop some feature not all pixel
        if len(inputs.shape) > 3:
            randn = np.random.rand(inputs.shape[0], inputs.shape[1])[:, :, np.newaxis, np.newaxis]
        elif len(inputs.shape) == 3:
            randn = np.random.rand(inputs.shape[0], inputs.shape[1])[:, :, np.newaxis]
        elif len(inputs.shape) == 2:
            randn = np.random.rand(inputs.shape[0], inputs.shape[1])
        self.mask = randn > self.dropout_probability
        output = self.inputs * self.mask
        # inverted dropout
        output = output / (1 - self.dropout_probability)
        return output
    
    def backward(self, delta, lr = ''):
        # #previous layer delta
        next_delta = (delta * self.mask) / (1 - self.dropout_probability)
        return next_delta


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

def one_hot_decoding(y_pred):
    return np.argmax(y_pred, axis=1)


if __name__ == '__main__':
    init()
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




# 定义神经网络
class MLP:
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


from conv2d_new import *

class ResNet9:
    def __init__(self, num_classes):
        # self.block1 = ConvBlock(inchannels=1, outchannels=64)
        # self.block2 = ConvBlock(inchannels=64, outchannels=128)
        # self.block34 = ResidualBlock(filters=128)
        # self.block5 = ConvBlock(inchannels=128, outchannels=256)
        # self.block6 = ConvBlock(inchannels=256, outchannels=512)
        # self.block78 = ResidualBlock(filters=512)  # 增加通道数
        # self.block9 = ConvBlock(inchannels=512*1, outchannels=num_classes, kernel_size=1, stride=1, padding=0)
        # channels_list = []

        self.block1 = ConvBlock(inchannels=1, outchannels=4)
        self.block2 = ConvBlock(inchannels=4, outchannels=16)
        self.block34 = ResidualBlock(filters=16)
        # self.block_avgpool1 = AvgPooling2D(kernel_size=(2, 2), stride=2)
        self.pool1 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        self.block5 = ConvBlock(inchannels=16, outchannels=32)
        self.block6 = ConvBlock(inchannels=32, outchannels=64)
        self.block78 = ResidualBlock(filters=64)  # 增加通道数
        # self.block_avgpool = AvgPooling2D(kernel_size=(4, 4), stride=4)
        # self.block_avgpool2 = AvgPooling2D(kernel_size=(2, 2), stride=2)
        self.pool2 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        # self.block9 = ConvBlock(inchannels=64*14*14, outchannels=num_classes, kernel_size=1, stride=1, padding=0)
        self.block9 = ConvBlock(inchannels=64*7*7, outchannels=num_classes, kernel_size=1, stride=1, padding=0)
        self.dropout = Dropout(0.9)
        # self.fc = np.random.randn(7 * 7 * 32, num_classes) * 0.01  # 最终卷积后输出
        # self.fc_b = np.zeros((1, num_classes))

    def forward(self, X, training=False):
        self.out1 = relu(self.block1(X))
        self.out2 = relu(self.block2(self.out1))
        self.out34 = relu(self.block34(self.out2))
        self.outavg1 = self.pool1(self.out34)
        self.out5 = relu(self.block5(self.outavg1))
        self.out6 = relu(self.block6(self.out5))
        self.out78 = relu(self.block78(self.out6))
        self.outavg2 = self.pool2(self.out78)
        out9 = self.outavg2.reshape(self.outavg2.shape[0], -1, 1, 1)
        if training:
            out9 = self.dropout.forward(out9)
        out9 = self.block9(out9)
        # out9 = self.block9(self.out78.reshape(self.out78.shape[0], -1, 1, 1))
        out9 = out9.reshape(out9.shape[0], -1)
        # out = out.reshape(out.shape[0], -1)  # 展平
        return softmax(out9)

    def backward(self, X, y_true, y_pred, learning_rate, weight_decay=1e-5):
        output_gradient = y_pred - y_true
        # output_gradient = np.dot(output_gradient, self.fc.T)
        output_gradient = output_gradient.reshape(output_gradient.shape[0], -1, 1, 1)
        output_gradient = self.block9.backward(output_gradient, learning_rate, weight_decay=weight_decay)

        output_gradient = self.dropout.backward(output_gradient)

        output_gradient = output_gradient.reshape(self.outavg2.shape)
        output_gradient = self.pool2.backward(output_gradient)

        # output_gradient = output_gradient.reshape(self.out78.shape)
        output_gradient = relu_backward(output_gradient, self.out78)
        output_gradient = self.block78.backward(output_gradient, learning_rate, weight_decay=weight_decay)

        output_gradient = relu_backward(output_gradient, self.out6)
        output_gradient = self.block6.backward(output_gradient, learning_rate, weight_decay=weight_decay)

        output_gradient = relu_backward(output_gradient, self.out5)
        output_gradient = self.block5.backward(output_gradient, learning_rate, weight_decay=weight_decay)

        output_gradient = self.pool1.backward(output_gradient)

        output_gradient = relu_backward(output_gradient, self.out34)
        output_gradient = self.block34.backward(output_gradient, learning_rate, weight_decay=weight_decay)

        output_gradient = relu_backward(output_gradient, self.out2)
        output_gradient = self.block2.backward(output_gradient, learning_rate, weight_decay=weight_decay)

        output_gradient = relu_backward(output_gradient, self.out1)
        output_gradient = self.block1.backward(output_gradient, learning_rate, weight_decay=weight_decay)

epochs = 200
epoch_test = 1
batch_size = 64
learning_rate = 1e-4
weight_decay = 0.
input_size = 784  # MNIST 输入大小 28x28
hidden_size = 64  # 隐藏层大小
output_size = 10  # 10 个类别
# nn = resnet9(input_size, hidden_size, output_size)
nn = ResNet9(num_classes=10)
# 训练神经网络
from tqdm import tqdm
loss_iter = np.zeros((epochs, 60000 // batch_size + 1))
loss_epoch = np.zeros((epochs, 1))
train_accuracy_epoch = np.zeros((epochs, 1))
test_accuracy_epoch = np.zeros((epochs, 1))

for epoch in range(epochs):
    # 随机打乱数据
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    
    # for i in range(0, x_train.shape[0], batch_size):
    for i in tqdm(range(0, x_train.shape[0], batch_size), leave=False, desc='train', ncols=0):
        x_batch = x_train[indices[i:i+batch_size]]
        y_batch = y_train_one_hot[indices[i:i+batch_size]]

        # 前向传播
        y_pred = nn.forward(x_batch.reshape(x_batch.shape[0], 1,28,28), training=True)

        # 计算损失
        loss = cross_entropy_loss(y_batch, y_pred)
        loss_iter[epoch, i//batch_size] = loss
        # print(loss)

        # 反向传播并更新权重
        # nn.backward(x_batch.reshape(x_batch.shape[0], 1,28,28), y_batch, y_pred, learning_rate * (0.5 ** (epoch // 20)) , weight_decay= weight_decay * (0.5 ** (epoch // 20)))
        nn.backward(x_batch.reshape(x_batch.shape[0], 1,28,28), y_batch, y_pred, learning_rate , weight_decay= weight_decay )
    
    # 每个 epoch 输出一次训练集的损失和准确率
    i = 0
    val_indices = np.arange(x_train.shape[0])
    x_batch = x_train[val_indices[i:i+batch_size]]
    train_pred = nn.forward(x_batch.reshape(x_batch.shape[0], 1,28,28))
    for i in tqdm(range(batch_size, x_train.shape[0], batch_size), leave=False, desc='train', ncols=0):
        x_batch = x_train[val_indices[i:i+batch_size]]
        train_pred = np.concatenate((train_pred, nn.forward(x_batch.reshape(x_batch.shape[0], 1,28,28))), axis=0)

    # train_pred = nn.forward(x_train.reshape(x_train.shape[0], 1,28,28))
    train_loss = cross_entropy_loss(y_train_one_hot, train_pred)
    train_acc = accuracy(y_train_one_hot, train_pred)
    loss_epoch[epoch] = train_loss
    train_accuracy_epoch[epoch] = train_acc
    
    log(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    if (epoch % epoch_test == 0) or (epoch == epochs - 1):
        i = 0
        test_indices = np.arange(x_test.shape[0])
        x_batch = x_test[test_indices[i:i+batch_size]]
        test_pred = nn.forward(x_batch.reshape(x_batch.shape[0], 1,28,28))
        for i in tqdm(range(batch_size, x_test.shape[0], batch_size), leave=False, desc='test', ncols=0):
            x_batch = x_test[test_indices[i:i+batch_size]]
            test_pred = np.concatenate((test_pred, nn.forward(x_batch.reshape(x_batch.shape[0], 1,28,28))), axis=0)
        # test_pred = nn.forward(x_test.reshape(x_test.shape[0], 1,28,28))
        test_acc = accuracy(y_test_one_hot, test_pred)
        test_accuracy_epoch[epoch] = test_acc
        log(f"Test Accuracy: {test_acc:.4f}")
        pred_indices = one_hot_decoding(test_pred).astype(np.uint8)
        label_pred_indices = np.concatenate((y_test[:,np.newaxis], pred_indices[:,np.newaxis]), axis=1) # 10000, 2

        with open(save_folder + os.sep+ f'checkpoint_epoch{epoch}.pkl', 'wb') as f:
            pickle.dump((nn, loss_iter, loss_epoch, train_accuracy_epoch, test_accuracy_epoch, label_pred_indices), f)