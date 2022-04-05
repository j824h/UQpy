import numpy as np

def Example1(samples=None):

    x = np.zeros(samples.shape[0])

    omega = 6.
    epsilon = 0.0001

    for i in range(samples.shape[0]):
        add = samples[i][1] - samples[i][0]*(omega+epsilon)**2
        diff = samples[i][0]*(omega-epsilon)**2 - samples[i][1]
        x[i] = np.maximum(add, diff)

    return x







def model_k(samples):
    return samples[0, 0] * samples[0, 1] - 80




def example2(samples=None):
    import numpy as np
    d = 2
    beta = 3.0902
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = -1 / np.sqrt(d) * (samples[i, 0] + samples[i, 1]) + beta
    return g


def example3(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = 6.2 * samples[i, 0] - samples[i, 1] * samples[i, 2] ** 2
    return g


def example4(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = samples[i, 0] * samples[i, 1] - 80
    return g


def RunPythonModel(samples, b_eff, d):

    qoi = list()
    for i in range(samples.shape[0]):
        qoi.append(b_eff * np.sqrt(d) - np.sum(samples[i, :]))
    return qoi

