import matplotlib.pyplot as plt
import cupy as np
import numpy
import torch
import pickle
# img = x_train[0,:].reshape(28,28) # First image in the training set.
# plt.imshow(img,cmap='gray')
# plt.show() # Show the image
from conv2d_new import ResidualBlock, ConvBlock


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

def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
        self.dropout = Dropout(0.99)
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

import torch.nn as nn
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

with open('../save/20241019/checkpoint_epoch67.pkl','rb') as f:
    np_checkpoint = pickle.load(f)


# pickle.dump((nn, loss_iter, loss_epoch, train_accuracy_epoch, test_accuracy_epoch, label_pred_indices), f)
with open('../save/20241019/checkpoint_epoch199.pkl','rb') as f:
    np_checkpoint2 = pickle.load(f)

torch_checkpoint = torch.load('../save/torch_20241019/checkpoint_epoch178.pt', weights_only=False)
torch_checkpoint2 = torch.load('../save/torch_20241019/checkpoint_epoch199.pt', weights_only=False)
# dict_keys(['model', 'loss_iter', 'loss_epoch', 'train_accuracy_epoch', 'test_accuracy_epoch', 'label_pred_indices'])

fig, axs = plt.subplots(3,2)
axs[0,0].plot(numpy.arange(200), np_checkpoint2[1].mean(axis=1).get())
axs[0,0].set_ylim(0,0.1)
axs[0,0].set_title('numpy train loss')
axs[0,1].plot(numpy.arange(200), torch_checkpoint2['loss_iter'].mean(dim=1,keepdim=False).numpy())
axs[0,0].set_ylim(0,0.1)
axs[0,1].set_title('torch train loss')

axs[1,0].plot(numpy.arange(200), np_checkpoint2[3].mean(axis=1).get())
axs[1,0].set_title('numpy train accuracy')
axs[1,0].set_ylim(0.96,1)
axs[1,1].plot(numpy.arange(200), torch_checkpoint2['train_accuracy_epoch'].mean(dim=1,keepdim=False).numpy())
axs[1,1].set_ylim(0.96,1)
axs[1,1].set_title('torch train accuracy')

np_test_accuracy_epoch = numpy.zeros(200)
torch_test_accuracy_epoch = numpy.zeros(200)
np_test_accuracy_epoch = np_checkpoint2[4].mean(axis=1).get()
torch_test_accuracy_epoch = torch_checkpoint2['test_accuracy_epoch'].mean(dim=1,keepdim=False).numpy()
# with open('../save/20241019/log.txt','r') as f:
#     aa = f.readlines()
# for ii in range(200):
#     np_test_accuracy_epoch[ii] = eval(aa[ii*2+1][len('Test Accuracy: '):-1])

# with open('../save/torch_20241012_AdamW/log.txt','r') as f:
#     aa = f.readlines()

# for ii in range(200):
#     torch_test_accuracy_epoch[ii] = eval(aa[ii*2+1][len('Test Accuracy: '):-1])

text_kwargs = dict(ha='center', va='center', fontsize=14, color='C1')
# axs[2,0].plot(numpy.arange(200), np_checkpoint2[1].mean(axis=1))
axs[2,0].plot(numpy.arange(200), np_test_accuracy_epoch)
axs[2,0].set_ylim(0.96,1)
axs[2,0].set_title('numpy test accuracy')
axs[2,0].text(67., np_test_accuracy_epoch[67], f'epoch68\n{np_test_accuracy_epoch[67]}', **text_kwargs)
# axs[2,1].plot(numpy.arange(200), torch_checkpoint2['test_accuracy_epoch'].mean(dim=1,keepdim=False).numpy())
axs[2,1].plot(numpy.arange(200), torch_test_accuracy_epoch)
axs[2,1].set_ylim(0.96,1)
axs[2,1].text(178., np_test_accuracy_epoch[178], 'epoch179\n%.4f'%torch_test_accuracy_epoch[178], **text_kwargs)
axs[2,1].set_title('torch test accuracy')
plt.show()


# fig2, axs2 = plt.subplots(3,1)
# wrong_indices = np.arange(10000)[np_checkpoint[5][:,0] != np_checkpoint[5][:,1]]
# with open("mnist.pkl",'rb') as f:
#     mnist = pickle.load(f)

# test_imgs, test_labels = mnist["test_images"], mnist["test_labels"]

# wrong_imgs = test_imgs[wrong_indices,:].reshape(wrong_indices.shape[0],28,28)
# wrong_imgs = wrong_imgs.transpose([1,0,2]).reshape(28,wrong_indices.shape[0] * 28)
# axs2[0].imshow(wrong_imgs.get(),cmap='gray')
# axs2[0].get_xaxis().set_visible(False)
# axs2[0].get_yaxis().set_visible(False)
# axs2[0].set_title('numpy failure case')
# text_kwargs = dict(ha='center', va='center', fontsize=20, color='C1')
# for ii in range(wrong_indices.shape[0]):
#     axs2[1].text(ii*28+14., 14, '%d'%np_checkpoint[5][wrong_indices[ii],1].get(), **text_kwargs)
# axs2[1].set_title('prediction')
# axs2[1].set_ylim(0.,28)
# axs2[1].set_xlim(0.,wrong_indices.shape[0]*28)
# axs2[1].get_xaxis().set_visible(False)
# axs2[1].get_yaxis().set_visible(False)
# axs2[2].set_title('label')
# for ii in range(wrong_indices.shape[0]):
#     axs2[2].text(ii*28+14., 14, '%d'%np_checkpoint[5][wrong_indices[ii],0].get(), **text_kwargs)
# axs2[2].get_xaxis().set_visible(False)
# axs2[2].get_yaxis().set_visible(False)
# axs2[2].set_ylim(0.,28)
# axs2[2].set_xlim(0.,wrong_indices.shape[0]*28)
# plt.show()


fig2, axs2 = plt.subplots(1,1)
wrong_indices = np.arange(10000)[np_checkpoint[5][:,0] != np_checkpoint[5][:,1]]
with open("mnist.pkl",'rb') as f:
    mnist = pickle.load(f)

test_imgs, test_labels = mnist["test_images"], mnist["test_labels"]

wrong_imgs = test_imgs[wrong_indices,:].reshape(wrong_indices.shape[0],28,28)
wrong_imgs = wrong_imgs.transpose([1,0,2]).reshape(28,wrong_indices.shape[0] * 28)
wrong_imgs = numpy.flip(wrong_imgs.get(),0)
axs2.imshow(wrong_imgs,cmap='gray')
axs2.get_xaxis().set_visible(False)
axs2.get_yaxis().set_visible(False)
axs2.set_title('numpy failure case')
axs2.set_ylim(-28*2.,28)
axs2.set_xlim(0.,wrong_indices.shape[0]*28)
text_kwargs = dict(ha='center', va='center', fontsize=20, color='C1')
for ii in range(wrong_indices.shape[0]):
    axs2.text(ii*28+14., -14, '%d'%np_checkpoint[5][wrong_indices[ii],1].get(), **text_kwargs)
# axs2[1].set_title('prediction')
# axs2[1].set_ylim(0.,28)
# axs2[1].set_xlim(0.,wrong_indices.shape[0]*28)
# axs2[1].get_xaxis().set_visible(False)
# axs2[1].get_yaxis().set_visible(False)
# axs2[2].set_title('label')
for ii in range(wrong_indices.shape[0]):
    axs2.text(ii*28+14., -28-14, '%d'%np_checkpoint[5][wrong_indices[ii],0].get(), **text_kwargs)
# axs2[2].get_xaxis().set_visible(False)
# axs2[2].get_yaxis().set_visible(False)
# axs2[2].set_ylim(0.,28)
# axs2[2].set_xlim(0.,wrong_indices.shape[0]*28)
plt.show()