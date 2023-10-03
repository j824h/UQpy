import pytest
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


@pytest.fixture
def srm_object():
    # User defined quantities
    number_of_times = 256
    number_of_frequencies = 128
    delta_time = 100 / number_of_times  # max_time / number_of_times
    delta_frequency = 1.28 / number_of_frequencies  # max_frequency / number_of_frequencies
    spectrum = lambda w: (125 / 4) * (w**2) * np.exp(-5 * w)

    # Computed quantities for SpectralRepresentation
    frequencies = np.linspace(0, (number_of_frequencies - 1) * delta_frequency, num=number_of_frequencies)
    power_spectrum = spectrum(frequencies)

    return SpectralRepresentation(power_spectrum,
                                  delta_time, delta_frequency,
                                  number_of_times, number_of_frequencies,
                                  random_state=128)


def test_samples_1d_1v_run_method_shape(srm_object):
    assert srm_object.samples is None
    number_of_samples = 100
    number_of_times = 256
    number_of_frequencies = 128
    srm_object.run(number_of_samples)
    assert srm_object.samples.shape == (number_of_samples, 1, number_of_times)
    assert srm_object.phi.shape == (number_of_samples, number_of_frequencies)


def test_samples_1d_1v_run_method_value(srm_object):
    assert srm_object.samples is None
    number_of_samples = 100
    srm_object.run(number_of_samples)
    assert np.isclose(srm_object.samples[53, 0, 134], -0.9143690244714813)
