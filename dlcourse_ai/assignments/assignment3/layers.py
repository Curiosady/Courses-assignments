import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")
    loss = reg_strength * np.sum(W * W)
    grad = reg_strength * 2 * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    preds = predictions.copy()
    preds -= np.max(preds, axis=1, keepdims=True)
    return np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    n_samples = probs.shape[0]
    return np.mean(-np.log(probs[np.arange(n_samples), target_index]))



def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probes = softmax(predictions)
    loss = cross_entropy_loss(probes, target_index)

    dprediction = probes.copy()
    n_samples = probes.shape[0]
    dprediction[np.arange(n_samples), target_index] -= 1
    dprediction /= n_samples

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.input = X.copy()
        return np.maximum(0, self.input)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_cur = self.input > 0
        return d_out * d_cur

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X[:, :]
        return self.X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_input = d_out.dot(self.W.value.T)

        self.W.grad += self.X.T.dot(d_out)
        self.B.grad += np.sum(d_out, axis=0).reshape(1, -1)

        return d_input


    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_pad = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, channels), X.dtype)
        self.X_pad[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X[:, :, :, :]

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        result = np.zeros((batch_size, out_height, out_width, self.out_channels), X.dtype)
        for y in range(0, out_height):
            for x in range(0, out_width):
                result[:, y, x, :] = self.X_pad[:, y:y + self.filter_size, x:x + self.filter_size, :].\
                    reshape(batch_size, -1).dot(self.W.value.reshape(-1, self.out_channels))
        result[:, :, :, :] += self.B.value
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X_pad.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_input = np.zeros_like(self.X_pad)
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(0, out_height):
            for x in range(0, out_width):
                self.W.grad += self.X_pad[:, y:y + self.filter_size, x:x + self.filter_size, :].\
                    reshape(batch_size, -1).T.dot(d_out[:, y, x, :].reshape(-1, out_channels)).reshape(self.W.value.shape)
                self.B.grad += d_out[:, y, x, :].sum(axis=0)
                d_input[:, y:y + self.filter_size, x:x + self.filter_size, :] += d_out[:, y, x, :].\
                    reshape(batch_size, -1).dot(self.W.value.reshape(-1, self.out_channels).T)\
                        .reshape(batch_size, self.filter_size, self.filter_size, channels)

        return d_input[:, self.padding:height - self.padding, self.padding:width - self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X[:, :, :, :]
        batch_size, height, width, channels = self.X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(0, out_height):
            for x in range(0, out_width):
                _sy = slice(y * self.stride, y * self.stride + self.pool_size)
                _sx = slice(x * self.stride, x * self.stride + self.pool_size)
                result[:, y, x, :] = self.X[:, _sy, _sx, :].reshape(batch_size, -1, channels).max(axis=1)
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        d_input = np.zeros_like(self.X)
        for y in range(0, out_height):
            for x in range(0, out_width):
                _sy = slice(y * self.stride, y * self.stride + self.pool_size)
                _sx = slice(x * self.stride, x * self.stride + self.pool_size)
                max_val = self.X[:, _sy, _sx, :].reshape(batch_size, -1, channels).max(axis=1, keepdims=True)
                mask = max_val == self.X[:, _sy, _sx, :].reshape(batch_size, -1, channels)
                new_val = np.tile(
                    d_out[:, y, x, :], (self.pool_size * self.pool_size)
                ).reshape(mask.shape) * mask
                d_input[:, _sy, _sx, :] += new_val.reshape(batch_size, self.pool_size, self.pool_size, channels)

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X = X[:, :, :, :]
        batch_size, height, width, channels = self.X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]W
        return self.X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X.shape)

    def params(self):
        # No params!
        return {}