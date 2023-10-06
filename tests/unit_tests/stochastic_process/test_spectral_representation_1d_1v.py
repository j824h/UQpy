import pytest
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


# Sample parameters
n_samples = 1
n_variables = 1
n_dimensions = 1
# Spectral Representation Parameters
n_times = 256
n_frequencies = 128
max_frequency = 12.8

frequency_interval = max_frequency / n_frequencies
frequencies = np.linspace(0, (n_frequencies - 1) * frequency_interval, num=n_frequencies)

max_time = 2 * np.pi / frequency_interval
time_interval = max_time / n_times

'''Power spectrum only has contribute at frequency 0.1 and 0.5, its a sum of two cosines'''
power_spectrum = np.zeros(shape=n_frequencies)
power_spectrum[1] = 1  # Has power at frequency 1 * frequency_interval = 0.1
power_spectrum[5] = 1  # Has power at frequency 100 * frequency_interval = 0.5
# power_spectrum = (frequencies ** 2) * np.exp(-frequencies)  # alternate, more complicated, power spectrum


@pytest.fixture
def srm_object():
    return SpectralRepresentation(power_spectrum,
                                  time_interval, frequency_interval,
                                  n_times, n_frequencies)


def sum_of_cosines(phi):
    """Computes the stochastic process as sum of cosines"""
    time = np.linspace(0, (n_times - 1) * time_interval, num=n_times)
    total = np.zeros(n_times)
    for i in range(n_frequencies):
        w_i = frequency_interval * i
        coefficient = 2 * np.sqrt(power_spectrum[i] * frequency_interval)
        term = coefficient * np.cos((w_i * time) + phi[0, i])
        total += term
    return total


def test_1d_1v_phi_zero(srm_object):
    """Test when all phi are zero"""
    phi = np.zeros(shape=(1, n_frequencies))
    srm_object.run(n_samples, phi=phi)
    cosines = sum_of_cosines(phi)
    assert srm_object.samples.shape == (n_samples, n_variables, n_times)
    assert all(np.isclose(srm_object.samples[0, 0, :], cosines))


def test_1d_1v_phi_constant(srm_object):
    """Test when all phi are constant"""
    phi = np.ones(shape=(1, n_frequencies)) * np.pi / 4
    srm_object.run(n_samples=n_samples, phi=phi)
    cosines = sum_of_cosines(phi)
    assert srm_object.samples.shape == (n_samples, n_variables, n_times)
    assert all(np.isclose(srm_object.samples[0, 0, :], cosines))


def test_1d_1v_phi_random_array(srm_object):
    """Test when all phi are random as defined by a random array"""
    phi = np.random.uniform(0, 2 * np.pi, size=(1, n_frequencies))
    srm_object.run(n_samples=10, phi=phi)
    cosines = sum_of_cosines(phi)
    assert srm_object.samples.shape == (n_samples, n_variables, n_times)
    assert all(np.isclose(srm_object.samples[0, 0, :], cosines))


def test_1d_1v_phi_random_state():
    """Test when all phase angles are random as defined by random_state and run method called via n_samples on init"""
    srm_object = SpectralRepresentation(power_spectrum,
                                        time_interval, frequency_interval,
                                        n_times, n_frequencies,
                                        n_samples=n_samples,
                                        random_state=123)
    cosines = sum_of_cosines(srm_object.phi)
    assert srm_object.phi.shape == (n_samples, n_frequencies)
    assert srm_object.samples.shape == (n_samples, n_variables, n_times)
    assert all(np.isclose(srm_object.samples[0, 0, :], cosines))
