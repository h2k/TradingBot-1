import numpy as np


def softmax(z):
    """
    Apply softmax on z matrix
    """
    assert len(z.shape) == 2
    max_s = np.max(z, axis=1)
    # Put back on two axis
    max_s = max_s[:, np.newaxis]
    e_x = np.exp(z - max_s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def get_state(parameters, t, window_size=20):
    """
    Get state of parameters in np array
    Also reshape them to fit our model
    """
    outside_list = []
    d = t - window_size + 1
    for parameter in parameters:
        blocking = (parameter[d:t + 1] if d >= 0 else -d * [parameter[0]] + parameter[0:t + 1])
        res = []
        for i in range(window_size - 1):
            res.append(blocking[i + 1] - blocking[i])
        for i in range(1, window_size, 1):
            res.append(blocking[i] - blocking[0])
        outside_list.append(res)
    return np.array(outside_list).reshape((1, -1))
