import numpy as np
from .utils import sign


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
