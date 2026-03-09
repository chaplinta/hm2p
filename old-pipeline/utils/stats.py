import numpy as np
from scipy.stats import entropy, pearsonr, spearmanr
from collections import namedtuple

"""
From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8970296/ Zong et al 2022
Originally from Skaggs 1996 
"""
def info_zong(bin_probs, bin_ca_means, ca_mean, epsilon=1e-8):

    # Check if bin_probs is a pandas series
    if hasattr(bin_probs, 'values'):
        bin_probs = bin_probs.values
    if hasattr(bin_ca_means, 'values'):
        bin_ca_means = bin_ca_means.values

    bin_probs = bin_probs.flatten()
    bin_ca_means = bin_ca_means.flatten()

    nan_indexes = np.logical_or(np.isnan(bin_probs), np.isnan(bin_ca_means))
    bin_probs = bin_probs[~nan_indexes]
    bin_ca_means = bin_ca_means[~nan_indexes]

    ca_mean = ca_mean + epsilon
    bin_ca_means = bin_ca_means + epsilon

    info = np.sum(bin_probs * (bin_ca_means / ca_mean) * np.log2(bin_ca_means / ca_mean))

    return info

def info_voights(bin_probs, bin_event_probs):

    bin_probs = bin_probs.flatten()
    bin_event_probs = bin_event_probs.flatten()

    nan_indexes = np.logical_or(np.isnan(bin_probs), np.isnan(bin_event_probs))
    bin_probs = bin_probs[~nan_indexes]
    bin_event_probs = bin_event_probs[~nan_indexes]

    # Calculate Kullback-Leibler Divergence
    kl_div = entropy(bin_event_probs.flatten(), bin_probs.flatten())

    return kl_div


def cohend(d1, d2):
    # From https://machinelearningmastery.com/effect-size-measures-in-python/

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

def cohend_mv(u1, u2, s1, s2, n1, n2):
    # From https://machinelearningmastery.com/effect-size-measures-in-python/
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the effect size
    return (u1 - u2) / s


# Define a result type compatible with scipy.stats's SpearmanrResult
ResultType = namedtuple('ResultType', ['statistic', 'pvalue'])

def nanpearsonr(x, y):

    x = x.flatten()
    y = y.flatten()

    nan_indexes = np.logical_or(np.isnan(x), np.isnan(y))
    x = x[~nan_indexes]
    y = y[~nan_indexes]

    if x.shape[0] == 0 or y.shape[0] == 0:
        return ResultType(np.nan, np.nan)

    if len(set(x)) == 1 or len(set(y)) == 1:
        return ResultType(np.nan, np.nan)

    return pearsonr(x, y)


def nanspearmanr(x, y):

    x = x.flatten()
    y = y.flatten()

    nan_indexes = np.logical_or(np.isnan(x), np.isnan(y))
    x = x[~nan_indexes]
    y = y[~nan_indexes]

    if x.shape[0] == 0 or y.shape[0] == 0:
        return ResultType(np.nan, np.nan)

    if len(set(x)) == 1 or len(set(y)) == 1:
        return ResultType(np.nan, np.nan)

    return spearmanr(x, y)

def calc_selectivity_index(x, y):
    result = (x - y) / (x + y)
    return result