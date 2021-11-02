from dataclasses import dataclass

from UQpy.distributions.baseclass import Distribution
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class MhInput(SamplingInput):
    """
    Class for providing input arguments to the :class:`.MetropolisHastings` class.

    :param callable pdf_target: Target density function from which to draw random samples. Either pdf_target or
     log_pdf_target must be provided (the latter should be preferred for better numerical stability).

     If pdf_target is a callable, it refers to the joint pdf to sample from, it must take at least one input x, which
     are the point(s) at which to evaluate the pdf. Within MCMC the pdf_target is evaluated as:
     p(x) = pdf_target(x, \*args_target)

     where x is a ndarray of shape (nsamples, dimension) and args_target are additional positional arguments that are
     provided to MCMC via its args_target input.

     If pdf_target is a list of callables, it refers to independent marginals to sample from. The marginal in dimension
     j is evaluated as: p_j(xj) = pdf_target[j](xj, \*args_target[j]) where x is a ndarray of shape (nsamples, dimension)
    :param callable log_pdf_target: Logarithm of the target density function from which to draw random samples.
     Either pdf_target or log_pdf_target must be provided (the latter should be preferred for better numerical
     stability).

     Same comments as for input pdf_target.
    :param tuple args_target: Positional arguments of the pdf / log-pdf target function. See pdf_target
    :param int burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
     no thinning during burn-in). Default is 0, no burn-in.
    :param int jump: Thinning parameter, used to reduce correlation between samples. Setting jump=n corresponds to
     skipping n-1 states between accepted states of the chain. Default is 1 (no thinning).
    :param int dimension: A scalar value defining the dimension of target density function. Either dimension and
     nchains or seed must be provided.
    :param list seed: Seed of the Markov chain(s), shape (nchains, dimension). Default: zeros(nchains x dimension).

     If seed is not provided, both nchains and dimension must be provided.
    :param bool save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
     Default: False
    :param bool concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
     are stored as an ndarray of shape (nsamples * nchains, dimension) if True, (nsamples, nchains, dimension) if False.
     Default: True
    :param int chains_number: The number of Markov chains to generate. Either dimension and nchains or seed must be
     provided.
    :param Distribution proposal: Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard
     multivariate normal
    :param bool proposal_is_symmetric: Indicates whether the proposal distribution is symmetric, affects computation of
     acceptance probability alpha Default: False, set to True if default proposal is used
    :param RandomStateType random_state: Random seed used to initialize the pseudo-random number generator. Default is
     None.

     If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
     object itself can be passed directly.
    """
    pdf_target: callable = None
    log_pdf_target: callable = None
    args_target: tuple = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: int = 1
    dimension: int = None
    seed: list = None
    save_log_pdf: bool = False
    concatenate_chains: bool = True
    chains_number: int = None
    proposal: Distribution = None
    proposal_is_symmetric: bool = False
    random_state: RandomStateType = None
