import numpy as np
import math as m

def extended_hist(
    x: np.ndarray,
    nbins: int,
    range: tuple[float, float],
    underflow: bool = True,
    overflow: bool = True,
    weights = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Histogram weighted data with potential under/overflow.

    Parameters
    ----------
    x : array_like
        Data to histogram.
    nbins : int
        Total number of bins.
    range : (float, float)
        Definition of binning max and min.
    underflow : bool
        Include undeflow data in the first bin.
    overflow : bool
        Include overflow data in the last bin.
    weights : array_like, optional
        Weights associated with each element of ``x``.

    Returns
    -------
    numpy.ndarray
        Total bin values.
    numpy.ndarray
        Poisson uncertainty on each bin count.
    numpy.ndarray
        Bin centers.
    numpy.ndarray
        Bin edges.

    """
    if weights is not None:
        if weights.shape != x.shape:
            raise ValueError(
                "Unequal shapes x: {}; weights: {}".format(
                    x.shape, weights.shape
                )
            )
    xmin, xmax = range
    edges = np.linspace(xmin, xmax, nbins + 1)
    neginf = np.array([-np.inf], dtype=np.float32)
    posinf = np.array([np.inf], dtype=np.float32)
    bins = np.concatenate([neginf, edges, posinf])
    if weights is None:
        hist, bin_edges = np.histogram(x, bins=bins)
    else:
        hist, bin_edges = np.histogram(x, bins=bins, weights=weights)

    n = hist[1:-1]
    if underflow:
        n[0] += hist[0]
    if overflow:
        n[-1] += hist[-1]

    if weights is None:
        u = np.sqrt(n)
    else:
        bin_sumw2 = np.zeros(nbins + 2, dtype=np.float32)
        digits = np.digitize(x, edges)
        for i in range(nbins + 2):
            bin_sumw2[i] = np.sum(
                np.power(weights[np.where(digits == i)[0]], 2)
            )
        u = bin_sumw2[1:-1]
        if underflow:
            u[0] += bin_sumw2[0]
        if overflow:
            u[-1] += bin_sumw2[-1]
        u = np.sqrt(u)

    centers = np.delete(edges, [0]) - (np.ediff1d(edges) / 2.0)
    return n, u, centers, edges

import math as m

def computeErrorRatio(hN, hD, uN,uD):
    errs_sup = []
    errs_neg = []
    for i in range(len(hN)):
        err_sup = 0.
        err_neg = 0.
        err = m.sqrt((1./hD[i])**2*uN[i]**2 + (hN[i]/hD[i]**2)**2*uD[i]**2)
        if(hN[i]/hD[i] + err > 1):
            err_sup = 1 - hN[i]/hD[i]
        else:
            err_sup = err
        err_neg = err
        errs_sup.append(err_sup)
        errs_neg.append(err_neg)
    return [errs_neg,errs_sup]

