import numpy as np
from UQpy.utilities import *
from beartype import beartype
from beartype.vale import Is
from typing import Annotated, Union
from UQpy.utilities.ValidationTypes import NumericArrayLike, PositiveInteger


class SpectralRepresentation:
    @beartype
    def __init__(
            self,
            power_spectrum: NumericArrayLike,
            time_interval: NumericArrayLike,
            frequency_interval: NumericArrayLike,
            n_time_intervals: NumericArrayLike,
            n_frequency_intervals: NumericArrayLike,
            n_samples: PositiveInteger = None,
            random_state: RandomStateType = None,
            phi: np.ndarray = None  # FixMe: This should be a numpy array with a weird shape
    ):
        """A class to simulate stochastic processes from a given power spectrum density using the
        Spectral Representation Method :cite:`StochasticProcess2`.

        This class can simulate uni-variate, multi-variate, and multi-dimensional stochastic processes.
        The class uses Singular Value Decomposition, as opposed to Cholesky Decomposition, to ensure robust,
        near-positive definite multi-dimensional power spectra.
        This class checks if the criteria :math:`\Delta t \leq 2\pi / 2\omega_u` is met, and raises a
        :code:`RuntimeError` if the inequality is violated.

        :param n_samples: Number of samples of the stochastic process to be simulated.
         The :py:meth:`run` method is automatically called if :code:`n_samples` is provided.
         If :code:`n_samples` is not provided, then the :class:`.SpectralRepresentation` object is created
         but samples are not generated.  # FixMe: This behavior was not implemented in `run` method
        :param power_spectrum: The discretized power spectrum.

         * For uni-variate, one-dimensional processes `power_spectrum` will be :class:`list` or :class:`numpy.ndarray` of length `n_frequency_intervals`.

         * For multi-variate, one-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_of_variables, n_variables, n_frequency_intervals)`.

         * For uni-variate, multi-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1])`

         * For multi-variate, multi-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_variables, n_variables, n_frequency_intervals[0],...,n_frequency_intervals[n_dimensions-1])`.

        :param time_interval: Length of time discretizations
         (:math:`\Delta t`) for each dimension of size :code:`n_dimensions`.
        :param frequency_interval: Length of frequency discretizations
         (:math:`\Delta \omega`) for each dimension of size :code:`n_dimensions`.
        :param n_time_intervals: Number of time discretizations
         for each dimensions of size :code:`n_dimensions`.
        :param n_frequency_intervals: Number of frequency discretizations
         for each dimension of size :code:`n_dimensions`.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :code:`None`.
         If an :code:`int` or :code:`np.random.RandomState` is provided, this sets :py:meth:`np.random.seed`.
        :param phi: Optional, phase angles used in the Spectral Representation Method.
        """
        self.power_spectrum = power_spectrum
        # if (
        #         isinstance(time_interval, float)
        #         and isinstance(frequency_interval, float)
        #         and isinstance(n_time_intervals, int)
        #         and isinstance(n_frequency_intervals, int)
        # ):
        #     time_interval = [time_interval]
        #     frequency_interval = [frequency_interval]
        #     n_time_intervals = [n_time_intervals]
        #     n_frequency_intervals = [n_frequency_intervals]
        # self.time_interval = np.array(time_interval)
        # self.frequency_interval = np.array(frequency_interval)
        # self.number_time_intervals = np.array(n_time_intervals)
        # self.number_frequency_intervals = np.array(n_frequency_intervals)
        self.time_interval = np.atleast_1d(time_interval)
        self.frequency_interval = np.atleast_1d(frequency_interval)
        self.number_time_intervals = np.atleast_1d(n_time_intervals)
        self.number_frequency_intervals = np.atleast_1d(n_frequency_intervals)
        self.n_samples = n_samples
        self.phi = phi
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        # Check if Equation 45 from Shinozuka and Deodatis 1991 is satisfied
        frequency_cutoff = self.frequency_interval * self.number_frequency_intervals
        max_time_interval = 2 * np.pi / (2 * frequency_cutoff)
        if (self.time_interval > max_time_interval).any():
            raise RuntimeError("UQpy: time_interval greater than pi / cutoff_frequency."
                               "Aliasing might occur during execution.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize Attributes
        self.samples: NumpyFloatArray = None
        """Generated samples.
        The shape of the samples is :code:`(n_samples, n_variables, n_time_intervals[0], ..., 
        n_time_intervals[n_dimensions-1])`"""
        # self.frequency = None  # Todo: How do we set this in the multidimensional, multivariate case
        # self.time = None # ToDo: How do we save time in multidimensional, multivariate case?
        self.n_variables: int = None
        """Number of variables in the stochastic process."""
        self.n_dimensions: int = len(self.number_frequency_intervals)
        """The dimensionality of the stochastic process."""
        # self.phi: NumpyFloatArray = None
        self.phi = phi  # FixMe: Double check the shape of the phi array
        """The random phase angles used in the simulation of the stochastic process.
        The shape of the phase angles :code:`(n_samples, n_variables, n_frequency_intervals[0], ...,
        n_frequency_intervals[n_dimensions-1])`"""

        if self.n_dimensions == len(self.power_spectrum.shape):
            self.case = "uni"
        else:
            self.n_variables = self.power_spectrum.shape[0]
            self.case = "multi"

        if self.n_samples is not None:  # Run Spectral Representation Method if n_samples provided
            self.run(n_samples=self.n_samples)

    @beartype
    def run(self, n_samples: PositiveInteger):  # ToDo: should you be able to put phi in here too?
        """Execute the random sampling in the :class:`.SpectralRepresentation` class.

        The :meth:`run` method is the function that performs random sampling in the :class:`.SpectralRepresentation`
        class. If `n_samples` is provided when the :class:`.SpectralRepresentation` object is defined, the
        :meth:`run` method is automatically called. The user may also call the :meth:`run` method directly to generate
        samples. The :meth:`run` method of the :class:`.SpectralRepresentation` class can be invoked many times and each
        time the generated samples are appended to the existing samples.

        :param n_samples: Number of samples of the stochastic process to be simulated.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` attribute of
        the :class:`.SpectralRepresentation` class.
        """
        self.logger.info("UQpy: Stochastic Process: Running Spectral Representation Method.")
        self.n_samples = n_samples  # ToDo: Is assigning this to the class attribute the best way to do this?
        phi = self.phi

        if self.case == "uni":
            self.logger.info("UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.")
            self.logger.info("UQpy: The number of dimensions is %i:", self.n_dimensions)
            if phi is None:
                size = np.append(self.n_samples, self.number_frequency_intervals)
                phi = np.random.uniform(low=0, high=2 * np.pi, size=size)
                # phi = (np.random.uniform(size=np.append(self.n_samples, np.ones(self.n_dimensions,
                #                                                                 dtype=np.int32)
                #                                         * self.number_frequency_intervals, )) * 2 * np.pi)
            samples = self._simulate_uni(phi)
        elif self.case == "multi":
            self.logger.info("UQpy: Stochastic Process: Starting simulation of multi-variate Stochastic Processes.")
            self.logger.info("UQpy: Stochastic Process: The number of variables is %i:", self.n_variables)
            self.logger.info("UQpy: Stochastic Process: The number of dimensions is  %i:", self.n_dimensions)
            if phi is None:
                size = np.append(self.n_samples, np.append(self.number_frequency_intervals, self.n_variables))
                phi = np.random.uniform(low=0, high=2 * np.pi, size=size)
                # phi = (np.random.uniform(size=
                #                          np.append(self.n_samples, np.append(np.ones(self.n_dimensions,
                #                                                                      dtype=np.int32)
                #                                                              * self.number_frequency_intervals,
                #                                                              self.n_variables, ), )) * 2 * np.pi)

            samples = self._simulate_multi(phi)

        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
        if self.phi is None:  # FixMe: this mistakenly concatenates when phi is not None
            self.phi = phi
        else:
            self.phi = np.concatenate((self.phi, phi), axis=0)

        self.logger.info("UQpy: Stochastic Process: Spectral Representation Method Complete.")

    def _simulate_uni(self, phi):
        """Simulate unidimensional spectral representation method using the phase angles :code:`phi`"""
        fourier_coefficient = np.exp(phi * 1.0j) * np.sqrt(2 ** (self.n_dimensions + 1)
                                                           * self.power_spectrum
                                                           * np.prod(self.frequency_interval)
                                                           )
        samples = np.fft.fftn(fourier_coefficient, s=self.number_time_intervals)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

    def _simulate_multi(self, phi):
        """Simulate multidimensional spectral representation method using the phase angles :code:`phi`"""
        power_spectrum = np.einsum("ij...->...ij", self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.n_dimensions + 1)) * np.sqrt(np.prod(self.frequency_interval))
        u, s, v = np.linalg.svd(power_spectrum)
        power_spectrum_decomposed = np.einsum("...ij,...j->...ij", u, np.sqrt(s))
        fourier_coefficient = coefficient * np.einsum("...ij,n...j -> n...i",
                                                      power_spectrum_decomposed,
                                                      np.exp(phi * 1.0j))
        fourier_coefficient[np.isnan(fourier_coefficient)] = 0
        samples = np.real(np.fft.fftn(fourier_coefficient,
                                      s=self.number_time_intervals,
                                      axes=tuple(np.arange(1, 1 + self.n_dimensions))))
        samples = np.einsum("n...m->nm...", samples)
        return samples
