import cupy as np
# import numpy as np
def im2col(X, filter_height, filter_width, stride=1, padding=0):
    n, h, w, c = X.shape
    out_height = (h + 2 * padding - filter_height) // stride + 1
    out_width = (w + 2 * padding - filter_width) // stride + 1

    # 添加填充
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    # 初始化列矩阵
    col = np.zeros((n, filter_height * filter_width * c, out_height * out_width))

    for i in range(out_height):
        for j in range(out_width):
            col[:, :, i * out_width + j] = X_padded[:, 
                i * stride:i * stride + filter_height, 
                j * stride:j * stride + filter_width, 
                :].reshape(n, -1)

    return col

def get_im2row_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    # 行坐标
    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i1 = np.repeat(np.arange(field_height), field_width)
    i1 = np.tile(i1, C)

    # 列坐标
    j0 = stride * np.tile(np.arange(out_width), out_height)
    j1 = np.tile(np.arange(field_width), field_height * C)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(1, -1)

    return (k, i, j)


def im2row_indices(x, field_height, field_width, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2row_indices(x.shape, field_height, field_width, padding, stride)

    rows = x_padded[:, k, i, j]
    C = x.shape[1]
    # 逐图像采集
    rows = rows.reshape(-1, field_height * field_width * C)
    return rows


def row2im_indices(rows, x_shape, field_height=3, field_width=3, padding=1, stride=1, isstinct=False):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=rows.dtype)
    k, i, j = get_im2row_indices(x_shape, field_height, field_width, padding,
                                 stride)
    rows_reshaped = rows.reshape(N, -1, C * field_height * field_width)
    np.add.at(x_padded, (slice(None), k, i, j), rows_reshaped)

    if isstinct:
        # 计算叠加倍数，恢复原图
        x_ones = np.ones(x_padded.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x_padded.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        x_padded = x_padded / x_zeros

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]

def conv_output2fc(inputs):
    output = inputs.copy()
    # [N, C, H, W]
    num, depth, height, width = output.shape[:4]

    # [N,C,H,W] —> [N,C,H*W]
    output = output.reshape(num, depth, -1)
    # [N,C,H*W] -> [N,H*W,C]
    output = output.transpose(0, 2, 1)
    # [N,H*W,C] -> [N*H*W,C]
    return output.reshape(-1, depth)

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    return dout * (x > 0)

class ResidualBlock:
    def __init__(self, filters, kernel_size=3, stride=1, padding=1):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.W1 = np.random.randn(3, 3, filters, filters) * 0.01  # 第一层卷积权重
        self.b1 = np.zeros((1, 1, 1, filters))  # 第一层卷积偏置
        self.W2 = np.random.randn(3, 3, filters, filters) * 0.01  # 第二层卷积权重
        self.b2 = np.zeros((1, 1, 1, filters))  # 第二层卷积偏置

    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, X):
        self.input = X
        self.A1 = relu(self.conv_forward(X, self.W1, self.b1))  # 计算第一层卷积的输出
        out2 = self.conv_forward(self.A1, self.W2, self.b2)  # 计算第二层卷积的输出
        self.output = out2 + self.input  # 残差连接
        return self.output

    def conv_forward(self, inputs, w, b):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out_w = int((W - self.kernel_size + 2 * self.padding) / self.stride + 1)

        a = im2row_indices(inputs, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding) # B*H*W, KKC
        # print(a.shape)
        w = w.reshape(-1, w.shape[-1])
        b = b.reshape(-1, b.shape[-1])
        z = a.dot(w) + b

        out = z.reshape(N, out_h, out_w, -1).transpose((0, 3, 1, 2))
        # self.cache = (a, inputs.shape, w, b)
        return out #, cache

    def backward(self, grad_out, learning_rate, weight_decay = 0.):

        a = im2row_indices(self.A1, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding) # B*H*W, KKCmid
        input_shape = self.A1.shape

        dz = conv_output2fc(grad_out) # [N*H*W,Cout]
        dW2 = a.T.dot(dz) # KKCin, Cout
        db2 = np.sum(dz, axis=0, keepdims=True) / dz.shape[0] # 1, Cout

        da = dz.dot(self.W2.reshape(-1, self.W2.shape[-1]).T) # [N*H*W,KKCmid]

        dX = row2im_indices(da, input_shape, field_height=self.kernel_size,
                                              field_width=self.kernel_size, stride=self.stride, padding=self.padding) # [N,Cmid,H,W]

        # 更新权重和偏置
        # self.W2 -= learning_rate * dW2.reshape(self.W2.shape)
        # self.b2 -= learning_rate * db2.reshape(self.b2.shape)
        self.W2 = (1-weight_decay) * self.W2 - learning_rate * dW2.reshape(self.W2.shape)
        self.b2 = (1-weight_decay) * self.b2 - learning_rate * db2.reshape(self.b2.shape)


        a = im2row_indices(self.input, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding) # B*H*W, KKCin
        input_shape = self.input.shape

        dz = conv_output2fc(dX) # [N*H*W,Cmid]
        dW1 = a.T.dot(dz) # KKCin, Cmid
        db1 = np.sum(dz, axis=0, keepdims=True) / dz.shape[0] # 1, Cmid

        da = dz.dot(self.W1.reshape(-1, self.W1.shape[-1]).T) # [N*H*W,KKCin]

        dX = row2im_indices(da, input_shape, field_height=self.kernel_size,
                                              field_width=self.kernel_size, stride=self.stride, padding=self.padding) # [N,Cin,H,W]
        # 更新权重和偏置
        # self.W1 -= learning_rate * dW1.reshape(self.W1.shape)
        # self.b1 -= learning_rate * db1.reshape(self.b1.shape)
        self.W1 = (1-weight_decay) * self.W1 - learning_rate * dW1.reshape(self.W1.shape)
        self.b1 = (1-weight_decay) * self.b1 - learning_rate * db1.reshape(self.b1.shape)
        return dX + grad_out  # 残差部分的梯度


        # # 反向传播到第二层卷积
        # dZ2 = output_gradient
        # dW2 = np.tensordot(self.A1.reshape(-1, self.A1.shape[-1]).T, dZ2.reshape(-1, dZ2.shape[-1]), axes=([0], [0]))
        # db2 = np.sum(dZ2, axis=0, keepdims=True)

        # # 反向传播到第一层卷积
        # dA1 = np.dot(dZ2.reshape(-1, dZ2.shape[-1]), self.W2.reshape(-1, self.W2.shape[-1]).T).reshape(self.A1.shape)
        # dA1[self.A1 <= 0] = 0  # ReLU 的导数

        # # 对于第一层的权重更新
        # col_input = im2col(self.input, 3, 3)  # 输入转换为列
        # dW1 = np.tensordot(col_input.transpose(0, 2, 1), dA1.reshape(-1, dA1.shape[-1]), axes=([0], [0]))
        # db1 = np.sum(dA1, axis=0, keepdims=True)

        # # 更新权重和偏置
        # self.W2 -= learning_rate * dW2
        # self.b2 -= learning_rate * db2
        # self.W1 -= learning_rate * dW1
        # self.b1 -= learning_rate * db1

        # # 返回输入的梯度
        # dZ1 = np.dot(dA1.reshape(-1, dA1.shape[-1]), self.W2.reshape(-1, self.W2.shape[-1]).T).reshape(self.input.shape)
        # return dZ1 + (output_gradient if self.input.shape == dZ2.shape else 0)  # 残差部分的梯度


class ConvBlock:
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1):
        # self.filters = filters
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.W = np.random.randn(kernel_size, kernel_size, inchannels, outchannels) * 0.01  # 卷积权重
        self.b = np.zeros((1, 1, 1, outchannels))  # 卷积偏置

    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, X):
        self.input = X
        # self.out = relu(self.conv_forward(X, self.W, self.b))  # 卷积 + ReLU
        self.out = self.conv_forward(X, self.W, self.b)  # 卷积 + ReLU
        return self.out

    # def conv_forward(self, X, W, b):
    def conv_forward(self, inputs, w, b):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out_w = int((W - self.kernel_size + 2 * self.padding) / self.stride + 1)

        a = im2row_indices(inputs, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding) # B*H*W, KKC
        # print(a.shape)
        w = w.reshape(-1, w.shape[-1])
        b = b.reshape(-1, b.shape[-1])
        z = a.dot(w) + b

        out = z.reshape(N, out_h, out_w, -1).transpose((0, 3, 1, 2))
        # self.cache = (a, inputs.shape, w, b)
        return out #, cache
    
    # def backward(self, output_gradient, learning_rate):
    def backward(self, grad_out, learning_rate, weight_decay=0.):
        assert len(grad_out.shape) == 4

        # a, input_shape, w, b = self.cache
        a = im2row_indices(self.input, self.kernel_size, self.kernel_size, stride=self.stride, padding=self.padding) # B*H*W, KKCin
        input_shape = self.input.shape

        dz = conv_output2fc(grad_out) # [N*H*W,Cout]
        dW = a.T.dot(dz) # KKCin, Cout
        db = np.sum(dz, axis=0, keepdims=True) / dz.shape[0] # 1, Cout

        da = dz.dot(self.W.reshape(-1, self.W.shape[-1]).T) # [N*H*W,KKCin]

        dX = row2im_indices(da, input_shape, field_height=self.kernel_size,
                                              field_width=self.kernel_size, stride=self.stride, padding=self.padding) # [N,Cin,H,W]

        # 更新权重和偏置
        self.W = (1-weight_decay) * self.W - learning_rate * dW.reshape(self.W.shape)
        self.b = (1-weight_decay) * self.b - learning_rate * db.reshape(self.b.shape)

        return dX


# 测试卷积层
if __name__ == "__main__":
    # 输入数据: batch_size=1, height=5, width=5, channels=1
    X = np.ones([1, 1, 5, 5])
    conv_layer = ConvBlock(inchannels=1, outchannels=2)
    res_layer  = ResidualBlock(filters=2)
    linear_layer  = ConvBlock(inchannels=2*5*5, outchannels=10)

    # 前向传播
    output = conv_layer.forward(X)
    output1 = relu(output)
    output = res_layer.forward(output1)
    output = output.reshape(output.shape[0], -1, 1, 1)
    output = linear_layer()
    print("Forward output shape:", output.shape)

    # 反向传播
    dout = np.random.randn(*output.shape)  # 假设来自下一层的梯度
    dX = res_layer.backward(dout, learning_rate=1e-4)
    dX = relu_backward(dX, output1)
    dX = conv_layer.backward(dX, learning_rate=1e-4)
    print("Backward output shape:", dX.shape)
