import numpy as np
from layers import FullyConnectedLayer, ReLULayer,\
   softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.FC1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.FC2 = FullyConnectedLayer(hidden_layer_size, n_output)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        params['W1'].grad = 0
        params['b1'].grad = 0
        params['W2'].grad = 0
        params['b2'].grad = 0

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        scores_fc1 = self.FC1.forward(X)
        scores_relu = self.ReLU.forward(scores_fc1)
        scores_fc2 = self.FC2.forward(scores_relu)

        loss, grad = softmax_with_cross_entropy(scores_fc2, y)
        
        grad_fc2 = self.FC2.backward(grad)
        grad_relu = self.ReLU.backward(grad_fc2)
        graf_fc1 = self.FC1.backward(grad_relu)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        W1_loss, W1_grad = l2_regularization(self.FC1.W.value, self.reg)
        self.FC1.W.grad += W1_grad
        B1_loss, B1_grad = l2_regularization(self.FC1.B.value, self.reg)
        self.FC1.B.grad += B1_grad
        W2_loss, W2_grad = l2_regularization(self.FC2.W.value, self.reg)
        self.FC2.W.grad += W2_grad
        B2_loss, B2_grad = l2_regularization(self.FC2.B.value, self.reg)
        self.FC2.B.grad += B2_grad
        self.params()
        loss = loss + W1_loss + B1_loss + W2_loss + B2_loss

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        scores_fc1 = self.FC1.forward(X)
        scores_relu = self.ReLU.forward(scores_fc1)
        scores_fc2 = self.FC2.forward(scores_relu)
        pred = np.argmax(scores_fc2, axis=1)

        return pred

    def params(self):
        result = {}
        result['W1'] = self.FC1.W
        result['b1'] = self.FC1.B
        result['W2'] = self.FC2.W
        result['b2'] = self.FC2.B

        return result
