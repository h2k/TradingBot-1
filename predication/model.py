import numpy as np


class Model(object):
    """
    AI Model for connection
    """
    def __init__(self, sZinput, sZlayer, sZoutput):
        """
        :param sZinput input size
        :param sZlayer layer size
        :param sZoutput output size
        """
        # Calculate the weight matrix
        # Denotes that input and layer size denotes as much
        self.weights = [
            np.random.rand(sZinput, sZlayer) * np.sqrt(1 / (sZinput + sZlayer)),
            np.random.rand(sZlayer, sZoutput) * np.sqrt(1 / (sZlayer + sZoutput)),
            np.zeros((1, sZlayer)), np.zeros((1, sZoutput)),
        ]

    def predict(self, inputs):
        feed_forward = np.dot(inputs, self.weights[0]) + self.weights[-2]
        return np.dot(feed_forward, self.weights[1]) + self.weights[-1]
