"""

Hyperbolic truncation and Least Angle Regression
======================================================================

In this example, we approximate the well-known Ishigami function with a total-degree Polynomial Chaos Expansion further
reduced by hyperbolic truncation. In order to reduce the number of basis functions, we use the best-model selection
algorithm based on Least Angle Regression.
"""

# %% md
#
# Import necessary libraries.

# %%

import numpy as np
import math
import numpy as np
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *


# %% md
#
# We then define the Ishigami function, which reads:
# :math:` f(x_1, x_2, x_3) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)`

# %%


def ishigami(xx):
    """Ishigami function"""
    a = 7
    b = 0.1
    term1 = np.sin(xx[0])
    term2 = a * np.sin(xx[1]) ** 2
    term3 = b * xx[2] ** 4 * np.sin(xx[0])
    return term1 + term2 + term3


# %% md
#
# The Ishigami function has three indepdent random inputs, which are uniformly distributed in
# interval :math:`[-\pi, \pi]`.

# %%

# input distributions
dist1 = Uniform(loc=-np.pi, scale=2 * np.pi)
dist2 = Uniform(loc=-np.pi, scale=2 * np.pi)
dist3 = Uniform(loc=-np.pi, scale=2 * np.pi)
marg = [dist1, dist2, dist3]
joint = JointIndependent(marginals=marg)

# %% md
#
# We now define our complete PCE, which will be further used for the best model selection algorithm.

# %%

# %% md
#
# We must now select a polynomial basis. Here we opt for a total-degree (TD) basis, such that the univariate
# polynomials have a maximum degree equal to :math:`P` and all multivariate polynomial have a total-degree
# (sum of degrees of corresponding univariate polynomials) at most equal to :math:`P`. The size of the basis
# is then given by
#
# .. math:: \frac{(N+P)!}{N! P!}
#
# where :math:`N` is the number of random inputs (here, :math:`N=3`).
#
# Note that the size of the basis is highly dependent both on :math:`N` and :math:`P:math:`. It is generally advisable
# that the experimental design has :math:`2-10` times more data points than the number of PCE polynomials. This might
# lead to curse of dimensionality and thus we will utilize the best model selection algorithm based on
# Least Angle Regression.

# %%

# maximum polynomial degree
P = 15
# construct total-degree polynomial basis
polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, P)

# %% md
#
# We must now compute the PCE coefficients. For that we first need a training sample of input random variable
# realizations and the corresponding model outputs. These two data sets form what is also known as an
# ''experimental design''. In case of adaptive construction of PCE by the best model selection algorithm, size of
# ED is given apriori and the most suitable basis functions are adaptively selected.

# %%

# create training data
sample_size = 500
print('Size of experimental design:', sample_size)

# realizations of random inputs
xx_train = joint.rvs(sample_size)
# corresponding model outputs
yy_train = np.array([ishigami(x) for x in xx_train])

# %% md
#
# We now fit the PCE coefficients by solving a regression problem. Here we opt for the _np.linalg.lstsq_ method,
# which is based on the _dgelsd_ solver of LAPACK. This original PCE class will be used for further selection of
# the best basis functions.

# %%

# fit model
least_squares = LeastSquareRegression()
pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
pce.fit(xx_train, yy_train)

# %% md
#
# Once we have created the PCE containing all basis functions generated by TD algorithm, it is possible to reduce
# the number of basis functions by LAR algorithm. The best model selection algorithm in UQPy is based on results
# of LAR adding basis functions to active set one-by-one until the target accuracy is obtained. Approximation
# error is measured by leave-one-out error :math:`Q^2` on given ED in :math:`[0,1]`. Target error represents the target
# accuracy measured by :math:`Q^2`.
#
# Note that if the target error is too high (close to 1), there is a risk of over-fitting. Therefore, we must check
# the over-fitting by empirical rule: if the three steps of LAR in row lead to decreasing accuracy - stop the
# algorithm. It is recommended to always check the over-fitting.

# %%

# check the size of the basis
print('Size of the full set of PCE basis:', polynomial_basis.polynomials_number)

target_error = 1
CheckOverfitting = True
pceLAR = polynomial_chaos.regressions.LeastAngleRegression.model_selection(pce, target_error, CheckOverfitting)

print('Size of the LAR PCE basis:', pceLAR.polynomial_basis.polynomials_number)

# %% md
#
# By simply post-processing the PCE's terms, we are able to get estimates regarding the mean and standard deviation of
# the model output.

# %%

mean_est, var_est = pceLAR.get_moments(higher=False)
print('PCE mean estimate:', mean_est)
print('PCE variance estimate:', var_est)

# %% md
#
# It is possible to obtain skewness and kurtosis (3rd and 4th moments), though it might be computationally demanding
# for high :math:`N` and :math:`P`.

# %%

mean_est, var_est, skew_est, kurt_est = pceLAR.get_moments(True)
print('PCE mean estimate:', mean_est)
print('PCE variance estimate:', var_est)
print('PCE skewness estimate:', skew_est)
print('PCE kurtosis estimate:', kurt_est)

# %% md
#
# Similarly to the statistical moments, we can very simply estimate the Sobol sensitivity indices, which quantify the
# importance of the input random variables in terms of impact on the model output.

# %%

from UQpy.sensitivity import *

pce_sensitivity = PceSensitivity(pceLAR)
pce_sensitivity.run()
sobol_first = pce_sensitivity.first_order_indices
sobol_total = pce_sensitivity.total_order_indices
print('First-order Sobol indices:')
print(sobol_first)
print('Total-order Sobol indices:')
print(sobol_total)

# %% md
#
# The accuracy of PCE is typically measured by leave-one-out error :math:`Q^2` on given ED. Moreover, we will test that
# also by computing the mean absolute error (MAE) between the PCE's predictions and the true model outputs, given a
# validation sample of :math:`10^5` data points.

# %%

n_samples_val = 100000
xx_val = joint.rvs(n_samples_val)
yy_val = np.array([ishigami(x) for x in xx_val])

yy_val_pce = pceLAR.predict(xx_val).flatten()
errors = np.abs(yy_val.flatten() - yy_val_pce)
MAE = (np.linalg.norm(errors, 1) / n_samples_val)

print('Mean absolute error:', MAE)
print('Leave-one-out cross validation on ED:', pceLAR.leaveoneout_error())

# %% md
#
# For the comparison, we can check results of PCE solved by OLS without the model selection algorithm. Note that, it is
# necessary to use 2−10 times more data points than the number of PCE polynomials.

# %%

# validation data sets
np.random.seed(999)  # fix random seed for reproducibility
n_samples_val = 100000
xx_val = joint.rvs(n_samples_val)
yy_val = np.array([ishigami(x) for x in xx_val])

mae = []  # to hold MAE for increasing polynomial degree
for degree in range(16):
    # define PCE
    polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, degree)
    least_squares = LeastSquareRegression()
    pce_metamodel = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)

    # create training data
    np.random.seed(1)  # fix random seed for reproducibility
    sample_size = int(pce_metamodel.polynomials_number * 5)
    xx_train = joint.rvs(sample_size)
    yy_train = np.array([ishigami(x) for x in xx_train])

    # fit PCE coefficients
    pce_metamodel.fit(xx_train, yy_train)

    # compute mean absolute validation error
    yy_val_pce = pce_metamodel.predict(xx_val).flatten()
    errors = np.abs(yy_val.flatten() - yy_val_pce)
    mae.append(np.linalg.norm(errors, 1) / n_samples_val)
    print('Size of ED:', sample_size)
    print('Polynomial degree:', degree)
    print('Mean absolute error:', mae[-1])
    print(' ')

# %% md
#
# In case of high-dimensional input and/or high :math:P` it is also beneficial to reduce the TD basis set by hyperbolic
# trunction. The hyperbolic truncation reduces higher-order interaction terms in dependence to parameter :math:`q` in
# interval :math:`(0,1)`. The set of multi indices :math:`\alpha` is reduced as follows:
#
# :math:`\alpha\in \mathbb{N}^{N}: || \boldsymbol{\alpha}||_q \equiv \Big( \sum_{i=1}^{N} \alpha_i^q \Big)^{1/q} \leq P`
#
# Note that :math:`q=1` leads to full TD set.

# %%

print('Size of the full set of PCE basis:', PolynomialBasis.create_total_degree_basis(joint, P).polynomials_number)
q = 0.8
polynomial_basis_hyperbolic = PolynomialBasis.create_total_degree_basis(joint, P, q)
# check the size of the basis
print('Size of the hyperbolic full set of PCE basis:', polynomial_basis_hyperbolic.polynomials_number)

# %% md
#
# The reduction of the full set size significantly reduces the necessary number of data points in ED for non-adaptive
# PCE. However, it is suitable only for mathematical models without significant higher-order interaction terms.

# %%

pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis_hyperbolic, regression_method=least_squares)
pce.fit(xx_train, yy_train)
yy_val_pce = pce.predict(xx_val).flatten()
errors = np.abs(yy_val.flatten() - yy_val_pce)
MAE = (np.linalg.norm(errors, 1) / n_samples_val)

print('Mean absolute error:', MAE)
print('Leave-one-out cross validation on ED:', pce.leaveoneout_error())
