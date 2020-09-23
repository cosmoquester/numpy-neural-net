import numpy as np
from .utils import sign
from collections import OrderedDict


def convolution2d(x, kernel, stride):
    """
    Convolution 2D : Do Convolution on 'x' with filter = 'kernel', stride = 'stride'

    [Input]
    x: 2D data (e.g. image)
    - Shape : (Height, Width)

    kernel : 2D convolution filter
    - Shape : (Kernel size, Kernel size)

    stride : Stride size
    - dtype : int

    [Output]
    conv_out : convolution result
    - Shape : (Conv_Height, Conv_Width)
    - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'Kernel size', 'Stride'
    """
    height, width = x.shape
    kernel_size = kernel.shape[0]
    conv_out = np.zeros(
        ((height - kernel_size) // stride + 1, (width - kernel_size) // stride + 1),
        dtype=np.float64,
    )

    for i in range(conv_out.shape[0]):
        for j in range(conv_out.shape[1]):
            conv_out[i, j] += np.sum(
                x[
                    i * stride : i * stride + kernel_size,
                    j * stride : j * stride + kernel_size,
                ]
                * kernel
            )

    return conv_out


class Perceptron:
    def __init__(self, num_features):
        self.W = np.random.rand(num_features, 1)
        self.b = np.random.rand(1)

    def forward(self, x):
        """
        Forward path of Perceptron (single neuron).
        x -- (weight, bias) --> z -- (sign function) --> sign(z)

        [Inputs]
            x : input for perceptron. Numpy array of (N, D)
        [Outputs]
            out : output of perceptron. Numpy array of (N, 1)g
        """
        if len(x.shape) < 2:
            x = np.expand_dims(x, 0)
        out = sign(np.matmul(x, self.W) + self.b)

        return out

    def stochastic_train(self, x, y, learning_rate):
        num_data = x.shape[0]
        """
        Stochastic Training of perceptron
        Perceptron Stochastic Training updates weights on every data (not batch)
        Training ends when there is no misclassified data.
        See Lecture Notes 'W09 Neural network basics'
        Again, Do not implement funtionalities such as shuffling data or sth.

        [Inputs]
            x : input for perceptron. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            learning_rate : learning rate.

        [Outputs]
            None
        """
        while True:
            # Repeat until quit condition is satisfied.
            is_end = True

            for i in range(num_data):
                y_hat = forward(x[i])
                if y_hat != y[i]:
                    is_end = False

                    for j in range(len(x[i])):
                        self.W += learning_rate * y[i] * x[i][j]
                        self.b += learning_rate * y[i]
            if is_end:
                break

    def batch_train(self, x, y, learning_rate):
        num_data = x.shape[0]
        """
        Batch Training of perceptron
        Perceptron Batch Training updates weights all at once for every data (not everytime)
        Training ends when there is no misclassified data.
        See Lecture Notes 'W09 Neural network basics'
        Again, Do not implement funtionalities such as shuffling data or sth.

        [Inputs]
            x : input for perceptron. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            learning_rate : learning rate.

        [Outputs]
            None
        """
        # gradients of W & b
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # Repeat until quit condition is satisfied.
        for i in range(num_data):
            # =============== EDIT HERE ===============
            y_hat = self.forward(x[i])

            if y_hat != y[i]:
                dW += np.expand_dims(y[i] * x[i], axis=0).T
                db += y[i]

            self.W += learning_rate * dW
            self.b += learning_rate * db


class ReLU:
    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.
    """

    def __init__(self):
        # 1 (True) if ReLU input < 0
        self.zero_mask = None

    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            out : ReLU(z).
        """
        out = z.copy()
        self.zero_mask = np.zeros_like(out)
        self.zero_mask[out >= 0] = 1
        out[out < 0] = 0

        return out

    def backward(self, d_prev):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients until now.
            d_prev = dL/dk, where k = ReLU(z).

        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        dz = d_prev * self.zero_mask
        return dz


class Sigmoid:
    """
    Sigmoid Function.
    Implement forward & backward path of Sigmoid.
    """

    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Sigmoid Forward.

        z --> (Sigmoid) --> self.out

        [Inputs]
            z : Sigmoid input in any shape.

        [Outputs]
            self.out : Sigmoid(z).
        """
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self, d_prev):
        """
        Sigmoid Backward.

        z --> (Sigmoid) --> self.out
        dz <-- (dSigmoid) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients until now.

        [Outputs]
            dz : Gradients w.r.t. Sigmoid input z.
        """
        dz = d_prev * self.out * (1 - self.out)
        return dz


class InputLayer:
    """
    Input Layer
    input -- (input layer) --> hidden

    Implement forward & backward path.
    """

    def __init__(self, num_features, num_hidden_1, activation):
        # Weights and bias
        self.W = np.random.rand(num_features, num_hidden_1)
        self.b = np.zeros(num_hidden_1)
        # Gradient of Weights and bias
        self.dW = None
        self.db = None
        # Forward input
        self.x = None
        # Activation function (Sigmoid or ReLU)
        self.act = activation()

    def forward(self, x):
        """
        Input layer forward
        - Feed forward
        - Apply activation function you implemented above.

        [Inputs]
           x : Input data (N, D)

        [Outputs]
            self.out : Output of input layer. Hidden. (N, H)
        """
        self.x = x
        self.out = self.act.forward(np.matmul(x, self.W) + self.b)
        return self.out

    def backward(self, d_prev):
        """
        Input layer backward
        x and (W & b) --> z -- (activation) --> hidden
        dx and (dW & db) <-- dz <-- (activation) <-- hidden

        - Backward of activation
        - Gradients of W, b

        [Inputs]
            d_prev : Gradients until now.

        [Outputs]
            None
        """
        back = self.act.backward(d_prev)
        self.db = np.sum(back, axis=0)
        self.dW = np.matmul(self.x.transpose(), back)


class SigmoidOutputLayer:
    def __init__(self, num_hidden_2, num_outputs):
        # Weights and bias
        self.W = np.random.rand(num_hidden_2, num_outputs)
        self.b = np.zeros(num_outputs)
        # Gradient of Weights and bias
        self.dW = None
        self.db = None
        # Input (x), label(y), prediction(y_hat)
        self.x = None
        self.y = None
        self.y_hat = None
        # Loss
        self.loss = None
        # Sigmoid function
        self.sigmoid = Sigmoid()

    def forward(self, x, y):
        """
        Sigmoid output layer forward
        - Make prediction
        - Calculate loss

        ## Already Implemented ##
        """
        self.y_hat = self.predict(x)
        self.y = y
        self.x = x

        self.loss = self.binary_ce_loss(self.y_hat, self.y)

        return self.loss

    def binary_ce_loss(self, y_hat, y):
        """
        Calcualte "Binary cross-entropy loss"
        Add 'eps' for stability inside log function.

        [Inputs]
            y_hat : Prediction
            y : Label

        [Outputs]
            loss value
        """
        eps = 1e-10
        bce_loss = -np.average(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

        return bce_loss

    def predict(self, x):
        """
        Make prediction in probability. (Not 0 or 1 label!!)

        [Inputs]
            x : input data

        [Outputs]
            y_hat : Prediction
        """
        z = np.matmul(x, self.W) + self.b
        y_hat = self.sigmoid.forward(z)

        return y_hat

    def backward(self, d_prev=1):
        """
        Calculate gradients of input (x), W, b of this layer.
        Save self.dW, self.db to update later.

        x and (W & b) --> z -- (activation) --> y_hat --> Loss
        dx and (dW & db) <-- dz <-- (activation) <-- dy_hat <-- Loss

        [Inputs]
            d_prev : Gradients until here. (Always 1 since its output layer)

        [Outputs]
            dx : Gradients of output layer input x (Not MLP input x!)
        """
        batch_size = self.y.shape[0]

        d_prev *= ((1 - self.y) / (1 - self.y_hat) - self.y / self.y_hat) / batch_size
        back = self.sigmoid.backward(d_prev)
        self.db = np.sum(back, axis=0)
        self.dW = np.sum(self.x * back, axis=0, keepdims=True).T
        dx = np.matmul(back, self.W.transpose())

        return dx


class HiddenLayer:
    def __init__(self, num_hidden_1, num_hidden_2):
        # Weights and bias
        self.W = np.random.rand(num_hidden_1, num_hidden_2)
        self.b = np.zeros(num_hidden_2)
        # Gradient of Weights and bias
        self.dW = None
        self.db = None
        # ReLU function
        self.act = ReLU()

    def forward(self, x):
        """
        Hidden layer forward
        - Feed forward
        - Apply activation function you implemented above.

        [Inputs]
           x : Input data (N, D)

        [Outputs]
            self.out : Output of input layer. Hidden. (N, H)
        """
        self.x = x
        self.out = self.act.forward(np.matmul(x, self.W) + self.b)

        return self.out

    def backward(self, d_prev):
        """
        Hidden layer backward
        x and (W & b) --> z -- (activation) --> output
        dx and (dW & db) <-- dz <-- (activation) <-- output

        - Calculate gradients of input (x), W, b of this layer.
        - Save self.dW, self.db to update later.

        [Inputs]
            d_prev : Gradients until here.

        [Outputs]
            dx : Gradients of output layer input x (Not MLP input x!)
        """
        back = self.act.backward(d_prev)
        self.db = np.sum(back, axis=0)
        self.dW = np.matmul(self.x.transpose(), back)
        dx = np.matmul(back, self.W.transpose())

        return dx


class SoftmaxOutputLayer:
    def __init__(self, num_hidden_2, num_outputs):
        # Weights and bias
        self.W = np.random.rand(num_hidden_2, num_outputs)
        self.b = np.zeros(num_outputs)
        # Gradient of Weights and bias
        self.dW = None
        self.db = None
        # Input (x), label(y), prediction(y_hat)
        self.x = None
        self.y = None
        self.y_hat = None
        # Loss
        self.loss = None

    def forward(self, x, y):
        """
        Softmax output layer forward
        - Make prediction
        - Calculate loss

        ## Already Implemented ##
        """
        self.y_hat = self.predict(x)
        self.y = y
        self.x = x

        self.loss = self.ce_loss(self.y_hat, self.y)

        return self.loss

    def ce_loss(self, y_hat, y):
        """
        Calcualte "Cross-entropy loss"
        Add 'eps' for stability inside log function.

        [Inputs]
            y_hat : Prediction
            y : Label

        [Outputs]
            loss value
        """
        eps = 1e-10

        y = np.argmax(y, axis=1)
        losses = -np.log(y_hat[range(y.shape[0]), y] + eps)
        ce_loss = np.average(losses)

        return ce_loss

    def predict(self, x):
        """
        Make prediction in probability. (Not 0, 1, 2 ... label!!)
        # Use softmax function above.

        [Inputs]
            x : input data

        [Outputs]
            y_hat : Prediction
        """
        fx = np.matmul(x, self.W) + self.b
        fx -= np.expand_dims(fx.max(axis=1), 1)
        expos = np.exp(fx)
        linesums = np.sum(expos, axis=1)
        y_hat = expos / linesums.reshape((linesums.shape[0], 1))

        return y_hat

    def backward(self, d_prev=1):
        """
        Calculate gradients of input (x), W, b of this layer.
        Save self.dW, self.db to update later.

        x and (W & b) --> z -- (activation) --> y_hat --> Loss
        dx and (dW & db) <-- dz <-- (activation) <-- dy_hat <-- Loss

        [Inputs]
            d_prev : Gradients until here. (Always 1 since its output layer)

        [Outputs]
            dx : Gradients of output layer input x (Not MLP input x!)
        """
        batch_size = self.y.shape[0]
        back = (self.y_hat - self.y) / batch_size

        self.db = np.sum(back, axis=0)
        self.dW = np.matmul(self.x.T, back)
        dx = np.matmul(back, self.W.T)

        return dx


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros(out_channels, dtype=np.float32)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        Convolution Layer Forward.

        [Input]
        x: 4-D input batch data
        - Shape : (Batch size, In Channel, Height, Width)

        [Output]
        conv_out : convolution result
        - Shape : (Conv_Height, Conv_Width)
        - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'Kernel size', 'Stride'
        """
        self.x = x
        batch_size, in_channel, _, _ = x.shape
        conv = self.convolution(x, self.W, self.b, self.stride, self.pad)
        self.output_shape = conv.shape

        return conv

    def convolution(self, x, kernel, bias=None, stride=1, pad=0):
        """
        Convolution Operation.
        Add bias if bias is not none

        Use
        variables --> self.W, self.b, self.stride, self.pad, self.kernel_size
        function --> convolution2d (what you already implemented above.)

        [Input]
        x: 4-D input batch data
        - Shape : (Batch size, In Channel, Height, Width)
        kernel: 4-D convolution filter
        - Shape : (Out Channel, In Channel, Kernel size, Kernel size)
        bias: 1-D bias
        - Shape : (Out Channel)
        - default : None
        stride : Stride size
        - dtype : int
        - default : 1
        pad: pad value, how much to pad
        - dtype : int
        - default : 0

        [Output]
        conv_out : convolution result
        - Shape : (Batch size, Out Channel, Conv_Height, Conv_Width)
        - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'Kernel size', 'Stride'
        """
        batch_size, in_channel, _, _ = x.shape

        if pad > 0:
            x = self.zero_pad(x, pad)

        _, _, height, width = x.shape
        out_channel, _, kernel_size, _ = kernel.shape
        assert x.shape[1] == kernel.shape[1]

        conv = np.zeros(
            (
                batch_size,
                out_channel,
                (height - kernel_size) // stride + 1,
                (width - kernel_size) // stride + 1,
            ),
            dtype=np.float64,
        )

        for n in range(batch_size):
            for oc in range(out_channel):
                for ic in range(in_channel):
                    conv[n, oc] += convolution2d(x[n, ic], kernel[oc, ic], stride)
                if type(bias) != type(None):
                    conv[n, oc] += bias[oc]

        return conv

    def backward(self, d_prev):
        """
        Convolution Layer Backward.
        Compute derivatives w.r.t x, W, b (self.x, self.W, self.b)

        [Input]
        d_prev: Gradients value so far in back-propagation process.

        [Output]
        self.dx : Gradient values of input x (self.x)
        - Shape : (Batch size, channel, Heigth, Width)
        """
        batch_size, in_channel, height, width = self.x.shape
        out_channel, _, kernel_size, _ = self.W.shape

        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)

        self.dW = np.zeros_like(self.W, dtype=np.float64)
        self.db = np.zeros_like(self.b, dtype=np.float64)
        dx = np.zeros_like(self.x, dtype=np.float64)

        # dW
        for n in range(batch_size):
            for oc in range(out_channel):
                for ic in range(in_channel):
                    self.dW[oc, ic] += convolution2d(self.x[n, ic], d_prev[n, oc], self.stride)

        # db
        for n in range(batch_size):
            for oc in range(out_channel):
                self.db[oc] += np.sum(d_prev[n, oc])

        # dx
        d_prev = self.zero_pad(d_prev, (dx.shape[2] - d_prev.shape[2] + kernel_size - 1) // 2)

        for n in range(batch_size):
            for oc in range(out_channel):
                for ic in range(in_channel):
                    dx[n, ic] += convolution2d(d_prev[n, oc], self.W[oc, ic].T[::-1].T[::-1], self.stride)

        return dx

    def zero_pad(self, x, pad):
        """
        Zero padding
        Given x and pad value, pad input 'x' around height & width.

        [Input]
        x: 4-D input batch data
        - Shape : (Batch size, In Channel, Height, Width)

        pad: pad value. how much to pad on one side.
        e.g. pad=2 => pad 2 zeros on left, right, up & down.

        [Output]
        padded_x : padded x
        - Shape : (Batch size, In Channel, Padded_Height, Padded_Width)
        """
        batch_size, in_channel, height, width = x.shape
        padded_x = np.zeros(
            (batch_size, in_channel, height + pad * 2, width + pad * 2),
            dtype=np.float64,
        )

        for n in range(batch_size):
            for ic in range(in_channel):
                padded_x[n, ic] = np.pad(x[n, ic], pad, "constant", constant_values=0)

        return padded_x


class MaxPoolingLayer:
    def __init__(self, kernel_size, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Max-Pooling Layer Forward. Pool maximum value by striding kernel.

        If image size is not divisible by pooling size (e.g. 4x4 image, 3x3 pool, stride=2),
        only pool from valid region, not go beyond the input image.
        4x4 image, 3x3 pool, stride=2 => 1x1 out
        (* Actually you should set kernel/pooling size, stride and pad properly, so that this does not happen.)

        [Input]
        x: 4-D input batch data
        - Shape : (Batch size, In Channel, Height, Width)

        [Output]
        pool_out : max_pool result
        - Shape : (Batch size, Out Channel, Pool_Height, Pool_Width)​
        - Pool_Height & Pool_Width can be calculated using 'Height', 'Width', 'Kernel size', 'Stride'
        """
        max_pool = None
        batch_size, channel, height, width = x.shape
        # Where it came from x. (1 if it is pooled, 0 otherwise.)
        # Might be useful when backward
        self.mask = np.zeros_like(x)

        kernel_size = self.kernel_size
        stride = self.stride
        max_pool = np.zeros(
            (
                batch_size,
                channel,
                (height - kernel_size) // stride + 1,
                (width - kernel_size) // stride + 1,
            ),
            dtype=float,
        )

        for n in range(batch_size):
            for c in range(channel):
                for hi, h in enumerate(range(kernel_size - 1, height, stride)):
                    for wi, w in enumerate(range(kernel_size - 1, width, stride)):
                        tmp = x[
                            n,
                            c,
                            h - kernel_size + 1 : h + 1,
                            w - kernel_size + 1 : w + 1,
                        ]
                        pld = np.unravel_index(tmp.argmax(), tmp.shape)
                        self.mask[
                            n,
                            c,
                            h - kernel_size + 1 + pld[0],
                            w - kernel_size + 1 + pld[1],
                        ] = 1
                        max_pool[n, c, hi, wi] = tmp[pld]

        self.output_shape = max_pool.shape
        return max_pool

    def backward(self, d_prev=1):
        """
        Max-Pooling Layer Backward.
        In backward pass, Max-pool distributes gradients to where it came from in forward pass.

        [Input]
        d_prev: Gradients value so far in back-propagation process.
        - Shape can be varies since either Conv. layer or FC-layer can follow.
            (Batch_size, Channel, Height, Width)
            or
            (Batch_size, FC Dimension)

        [Output]
        d_max : max_pool gradients
        - Shape : (batch_size, channel, height, width) - same shape as input x
        """
        d_max = None
        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        batch, channel, height, width = d_prev.shape

        kernel_size = self.kernel_size
        stride = self.stride
        d_max = self.mask.copy()

        for n in range(batch):
            for c in range(channel):
                for h in range(kernel_size, d_max.shape[2] + 1, stride):
                    for w in range(kernel_size, d_max.shape[3] + 1, stride):
                        d_max[n, c, h - kernel_size : h, w - kernel_size : w] *= d_prev[
                            n,
                            c,
                            (h - kernel_size) // stride,
                            (w - kernel_size) // stride,
                        ]

        return d_max


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        # Weight Initialization
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim / 2)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        """
        FC Layer Forward.
        Use variables : self.x, self.W, self.b

        [Input]
        x: Input features.
        - Shape : (Batch size, In Channel, Height, Width)
        or
        - Shape : (Batch size, input_dim)

        [Output]
        self.out : fc result
        - Shape : (Batch size, output_dim)
        """
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        self.x = x
        self.out = np.matmul(x, self.W) + self.b

        return self.out

    def backward(self, d_prev):
        """
        FC Layer Backward.
        Use variables : self.x, self.W

        [Input]
        d_prev: Gradients value so far in back-propagation process.

        [Output]
        dx : Gradients w.r.t input x
        - Shape : (batch_size, input_dim) - same shape as input x
        """
        self.dW = np.zeros_like(self.W, dtype=np.float64)  # Gradient w.r.t. weight (self.W)
        self.db = np.zeros_like(self.b, dtype=np.float64)  # Gradient w.r.t. bias (self.b)
        dx = np.zeros_like(self.x, dtype=np.float64)  # Gradient w.r.t. input x

        self.db = np.sum(d_prev, axis=0)
        self.dW = np.matmul(self.x.transpose(), d_prev)
        dx = np.matmul(d_prev, self.W.transpose())

        return dx


class SoftmaxLayer:
    def __init__(self):
        self.y = None
        self.y_hat = None

    def forward(self, x):
        """
        Softmax Layer Forward.
        Apply softmax (not log softmax or others...) on axis-1

        [Input]
        x: Score to apply softmax
        - Shape: (N, C)

        [Output]
        y_hat: Softmax probability distribution.
        - Shape: (N, C)
        """
        x -= np.expand_dims(x.max(axis=1), 1)
        expos = np.exp(x)
        linesums = np.sum(expos, axis=1)
        self.y_hat = expos / linesums.reshape((linesums.shape[0], 1))

        return self.y_hat

    def backward(self, d_prev=1):
        """

        Softmax Layer Backward.
        Gradients w.r.t input score.

        That is,
        Forward  : softmax prob = softmax(score)
        Backward : dL / dscore => 'dx'

        Compute dx (dL / dscore).
        Check loss function in HW5 pdf file.

        """
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) / batch_size

        return dx

    def ce_loss(self, y_hat, y):
        """

        Compute Cross-entropy Loss.
        Use epsilon (eps) for numerical stability in log.
        Epsilon 값을 계산의 안정성을 위해 log에 사용하세요.

        Check loss function in HW5 pdf file.
        Loss Function 을 과제 파일에서 확인하세요.

        [Input]
        y_hat: Probability after softmax.
        - Shape : (Batch_size, # of class)

        y: One-hot true label
        - Shape : (Batch_size, # of class)

        [Output]
        loss : cross-entropy loss
        - Single float

        """
        eps = 1e-10
        self.y_hat = y_hat
        self.y = y

        y = np.argmax(y, axis=1)
        losses = -np.log(y_hat[range(y.shape[0]), y] + eps)
        loss = np.average(losses)

        return loss
