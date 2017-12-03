from functools import partial
from SampleMethods import *
import matplotlib.pyplot as plt
from RunModel import RunModel


print()
''''
distribution
dimension
parameters
method
input_file
input_dir
save_format 
save_name=None
'''

'''
model = 'KO_1D'
output_type = 'scalar'
gfunction, handle = extract(model)


dimension = handle['dimension']
distribution = handle['distribution']
parameters = handle['parameters']
method = 'mcs'
nsamples = 130
lss = ['lss_parameters']

sm = SampleMethods(dimension=dimension, distribution=distribution, method=method, parameters=parameters)


# MCMC Code Block#
def normpdf(x):
    return stats.norm.pdf(x, 0, 1)


def mvnpdf(x):
    return stats.multivariate_normal.pdf(x, mean=np.zeros(d), cov=np.identity(d))


def marginal(x, mp):
    return stats.norm.pdf(x, mp[0], mp[1])


d = 2        # dimension
marginal_parameters = [[0, 1], [2, 5]]
Cov = np.ones(d)
x_start = 1.5*np.ones(d)
# MT = [['Normal', 0, 1], ['beta', 2, 2, 1, 1]]
mcmc = sm.MCMC(nsamples=10000, dim=d, x0=x_start, method='MMH', proposal='Normal', params=Cov, target=marginal,
               target_params=marginal_parameters, njump=10)

#plt.plot(mcmc.samples[:,0],mcmc.samples[:,1],'x')
# plt.hist(mcmc.samples,bins=50)
# plt.figure(2)
# stats.probplot(mcmc.samples, plot=plt)
#plt.show()

# Stratified Sampling Block
# SS = Strata(nstrata=[2, 3])
# # SS = strata(input_file='/Users/MichaelShields/Documents/GitHub/UQ_algorithms/examples/strata.txt')
# SS_samples = sm.STS(strata=SS)
# # samples = sm.mcs(10,model)
# # samples = Sample_Points(sm, number=nsamples, model=model, gfunction=gfunction, Type=output_type, interpreter='python')
# plt.plot(SS_samples.samples[:,0],SS_samples.samples[:,1],'x')
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.show()


#Subset Simulation Block
SuS = sm.SuS(nsamples=1000, dimension=2, p0=0.1, method='mMH', proposal='Uniform', width=2, model=sm.RunModel, fail=0)

print()
plt.plot(SuS.samples[0, :], SuS.samples[1, :],'x')

#plt.show()

'''

# MCMC Code Block


def normpdf(x):
    return stats.norm.pdf(x, 0, 1)


def mvnpdf(x):
    return stats.multivariate_normal.pdf(x, mean=np.zeros(2), cov=np.identity(2))


def marginal(x, mp):
    return stats.norm.pdf(x, mp[0], mp[1])


dimension = 2
marginal_parameters = [[0, 1], [2, 5]]
Cov = np.ones(dimension)
x_start = np.zeros(dimension)
distribution = ['Uniform', 'Uniform']
parameters = [[0, 1], [2, 3]]
model = partial(model_ko2d)

sm = SampleMethods(dimension=dimension, distribution=distribution, parameters=parameters)
# Monte Carlo Simulation Block #########################################################################################

mcs = sm.MCS(10, dimension=2)
g0 = RunModel(generator=sm, input=mcs.samples, model=model)

samples = '/Users/dimitrisgiovanis/Desktop/UQ_algorithms/util_/samples.txt'
g1 = RunModel(generator=sm, input=samples, model=model)


g2 = RunModel(generator=sm, nsamples=100, method='mcs', model=model)


# Stratified Sampling Block  ###########################################################################################

sts = sm.STS(strata=Strata(nstrata=[2, 3]))
f0 = RunModel(generator=sm, input=sts.samples, model=model)

#sts_ = sm.STS(strata=Strata(input_file='/Users/dimitrisgiovanis/Desktop/UQ_algorithms/util_/strata.txt'))
#f0_ = RunModel(generator=sm, input=sts_.samples, model=model)


f1 = RunModel(generator=sm, method='sts', model=model, sts_input=[2, 3])
#f1_ = RunModel(generator=sm, method='sts', model=model, sts_input='/Users/dimitrisgiovanis/Desktop/UQ_algorithms/util_/strata.txt')


# MCMC Block  ###########################################################################################

mcmc = sm.MCMC(nsamples=10000, dim=dimension, x0=x_start, method='MMH', proposal='Normal', params=Cov, target=marginal,
               target_params=marginal_parameters, njump=10)
plt.plot(mcmc.samples[:, 0], mcmc.samples[:, 1], 'x')
plt.show()
h0 = RunModel(generator=sm, input=mcmc.samples, model=model)
print()





print()



