import pytest
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


# Sample parameters
n_samples = 1
n_variables = 1
n_dimensions = 2
# Spectral Representation Parameters
n_times = np.array([256, 256])
n_frequencies = np.array([128, 128])
max_frequency = np.array([12.8, 6.4])

frequency_interval = max_frequency / n_frequencies
frequency_vectors = [np.linspace(0, (n_frequencies - 1) * frequency_interval[i], num=n_frequencies)
                     for i in range(n_dimensions)]
frequencies = np.meshgrid(*frequency_vectors, indexing='ij')

max_time = 2 * np.pi / frequency_interval
time_interval = max_time / n_times
x_vector = np.linspace(0, (n_times[0] - 1) * time_interval[0], num=n_frequencies[0])
y_vector = np.linspace(0, (n_times[1] - 1) * time_interval[1], num=n_frequencies[1])
coordinates = np.meshgrid(x_vector, y_vector)

size = (n_dimensions, n_frequencies[0], n_frequencies[1])
phi = np.zeros(size)


def power_spectrum(w_1, w_2):
    """Define n-dimension univariate power spectrum"""
    if w_1 == 0.5 and w_2 == 0:
        return 1
    # elif w_1 == 0 and w_2 == 1.5:
    #     return 1
    # elif w_1 == 0.4 and w_2 == 1.0:
    #     return 1
    else:
        return 0


def sum_of_cosines(x, y, phi):
    """Stochastic process defined using sum of cosines from Eq 44 of
    Simulation of multidimensional Gaussian stochastic fields by spectral representation (1996)"""
    total = 0
    for i in n_frequencies[0]:
        for j in n_frequencies[1]:
            kappa_1 = i * frequency_interval[0]
            kappa_2 = j * frequency_interval[1]
            coefficient_1 = np.sqrt(2 * power_spectrum(kappa_1, kappa_2) * np.prod(frequency_interval))
            coefficient_2 = np.sqrt(2 * power_spectrum(kappa_1, -kappa_2) * np.prod(frequency_interval))
            input_1 = (kappa_1 * x) + (kappa_2 * y) + phi[0, i, j]
            input_2 = (kappa_1 * x) - (kappa_2 * y) + phi[1, i, j]
            term = coefficient_1 * np.cos(input_1) + coefficient_2 * np.cos(input_2)
            total += term
    return total


@pytest.fixture
def srm_object():
    return SpectralRepresentation(power_spectrum(frequencies),
                                  time_interval,
                                  frequency_interval,
                                  n_times,
                                  n_frequencies)


def test_samples_nd_1v_shape(srm_object):
    srm_object.run(n_samples)
    assert srm_object.samples.shape == (n_samples, n_variables, n_times, n_times)


def test_nd_1v_values(srm_object):
    cosines = np.zeros(n_times)
    for i in range(n_times[0]):
        x = x_vector[i]
        for j in range(n_times[1]):
            y = y_vector[j]
            cosines[i, j] = sum_of_cosines(x, y, phi)
