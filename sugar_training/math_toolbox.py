"""math toolbox for sugar."""

from scipy import linalg, stats
import numpy as np
import copy
from astropy.stats import median_absolute_deviation as mad_astropy

def comp_rms(residuals, dof, err=True, variance=None):
    """
    Compute the RMS or WRMS of a given distribution.
    :param 1D-array residuals: the residuals of the fit.
    :param int dof: the number of degree of freedom of the fit.
    :param bool err: return the error on the RMS (WRMS) if set to True.
    :param 1D-aray variance: variance of each point. If given,
                             return the weighted RMS (WRMS).
    :return: rms or rms, rms_err
    """
    if variance is None: # RMS
        rms = float(np.sqrt(np.sum(residuals**2)/dof))
        rms_err = float(rms / np.sqrt(2.*dof))
    else: # Weighted RMS
        assert len(residuals) == len(variance)
        rms = float(np.sqrt(np.sum((residuals**2)/variance) / np.sum(1./variance)))
        rms_err = np.sqrt(2.*len(residuals)) / (2*np.sum(1./variance)*rms)
    if err:
        return rms, rms_err
    else:
        return rms

def svd_inverse(matrix,return_logdet=False):
    """
    svd method using scipy.
    """
    U,s,V = linalg.svd(matrix)
    Filtre = (s>10**-15)
    if np.sum(Filtre)!=len(Filtre):
        print('Pseudo-inverse decomposition :', len(Filtre)-np.sum(Filtre))

    inv_S = np.diag(1./s[Filtre])
    inv_matrix = np.dot(V.T[:,Filtre],np.dot(inv_S,U.T[Filtre]))

    if return_logdet:
        log_det=np.sum(np.log(s[Filtre]))
        return inv_matrix,log_det
    else:
        return inv_matrix

    
def cholesky_inverse(matrix,return_logdet=False):
    """
    Cholesky method using scipy.
    """
    L = linalg.cholesky(matrix, lower=True)
    inv_L = linalg.inv(L)
    inv_matrix = np.dot(inv_L.T, inv_L)

    if return_logdet:
        log_det=np.sum(2.*np.log(np.diag(L)))
        return inv_matrix,log_det
    else:
        return inv_matrix


def passage(norm_data1, norm_error1, vecteur_propre, sub_space=None):
    """
    project in emfa space.
    """
    norm_data=norm_data1.T
    norm_error=norm_error1.T

    if sub_space==None:
        Y=np.zeros(np.shape(norm_data))
        for sn in range(len(norm_data[0])):
            Y[:,sn]=np.dot(np.dot(np.linalg.inv(np.dot(vecteur_propre.T,np.dot(np.eye(len(norm_data))*(1./norm_error[:,sn]**2),vecteur_propre))),np.dot(vecteur_propre.T,np.eye(len(norm_data))*(1./norm_error[:,sn]**2))),norm_data[:,sn])

    else:
        Y=np.zeros((sub_space,len(norm_data[0])))
        vec_propre_sub_space=np.zeros((len(norm_data),sub_space))
        for vector in range(sub_space):
            vec_propre_sub_space[:,vector]=vecteur_propre[:,vector]
        for sn in range(len(norm_data[0])):
            Y[:,sn]=np.dot(np.dot(np.linalg.inv(np.dot(vec_propre_sub_space.T,np.dot(np.eye(len(norm_data))*(1./norm_error[:,sn]**2),vec_propre_sub_space))),np.dot(vec_propre_sub_space.T,np.eye(len(norm_data))*(1./norm_error[:,sn]**2))),norm_data[:,sn])

    return Y.T

def passage_plus_plus(data, cov, eig_vec, sub_space=None):
    """
    project in emfa space.
    """
    if sub_space is not None:
        vec = np.zeros((len(norm_data),sub_space))
        for vector in range(sub_space):
            vec[:,vector] = eig_vec[:,vector]
    else:
        vec = eig_vec
        sub_space = len(vec[0]) 

    projection = np.zeros((len(data[:,0]), sub_space))
    projection_cov = np.zeros((len(data[:,0]), sub_space, sub_space))

    for sn in range(len(data[:,0])):
        w = np.linalg.inv(cov[sn])
        A = np.linalg.inv(np.dot(vec.T, w.dot(vec)))

        projection[sn] = np.dot(A, np.dot(vec.T, w.dot(norm_data[:,sn])))
        projection_cov[sn] = A

    return projection, projection_cov



def passage_error(norm_error1,vecteur_propre,sub_space,return_std=False):
    """
    project error. 

    set return std to true if you want just the
    diagonal part.
    """
    norm_error=norm_error1.T

    error_new_base=np.zeros((len(norm_error[0]),sub_space))
    cov_Y=np.zeros((len(norm_error[0]),sub_space,sub_space))

    vec_propre_sub_space=np.zeros((len(norm_error),sub_space))
    for vector in range(sub_space):
        vec_propre_sub_space[:,vector]=vecteur_propre[:,vector]

    for sn in range(len(norm_error[0])):
        cov_Y[sn]=np.linalg.inv(np.dot(vec_propre_sub_space.T,np.dot(np.eye(len(norm_error))*(1./norm_error[:,sn]**2),vec_propre_sub_space)))
        error_new_base[sn]=np.sqrt(np.diag(cov_Y[sn]))

    if return_std:
        return error_new_base
    else:
        return cov_Y

def biweight_M(sample,CSTD=6.):
    """
    average using biweight (beers 90)
    """
    M = np.median(sample)
    iterate = [copy.deepcopy(M)]
    mu = (sample-M)/(CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = (sample-M)*((1.-mu**2)**2)
    down = (1.-mu**2)**2
    M = M + np.sum(up[Filtre])/np.sum(down[Filtre])
    
    iterate.append(copy.deepcopy(M))
    i=1
    while abs((iterate[i-1]-iterate[i])/iterate[i])<0.001:
    
        mu = (sample-M)/(CSTD*mad_astropy(sample))
        Filtre = (abs(mu)<1)
        up=(sample-M)*((1.-mu**2)**2)
        down = (1.-mu**2)**2
        M = M + np.sum(up[Filtre])/np.sum(down[Filtre])
        iterate.append(copy.deepcopy(M))
        i+=1
        if i == 100 :
            print('voila voila ')
            break
    return M
                                                                                                                                                                                                                                                    

def biweight_S(sample,CSTD=9.):
    """
    std using biweight (beers 90)
    """
    M = biweight_M(sample)
    mu = (sample-M)/(CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = ((sample-M)**2)*((1.-mu**2)**4)
    down = (1.-mu**2)*(1.-5.*mu**2)
    std = np.sqrt(len(sample))*(np.sqrt(np.sum(up[Filtre]))/abs(np.sum(down[Filtre])))
    return std#, mu

def loess(x, y, f=2./3., niter=3, madclip=6):
    """Fit a smooth non-parametric regression curve to a scatterplot.

    Loess smoother: Robust locally weighted regression.  The loess
    function fits a nonparametric regression curve to a scatterplot.
    The arrays *x* and *y* contain an equal number of elements; each
    pair (x[i], y[i]) defines a data point in the scatterplot. The
    function returns the estimated (smooth) values of *y*.

    The smoothing span is given by *f*. A larger value for *f* will
    result in a smoother curve. The number of robustifying iterations
    is given by *niter*. The function will run faster with a smaller
    number of iterations.

    :param 1D-array x: input 1D-array (*endogeneous*)
    :param 1D-array y: measured 1D-array (*exogeneous*)
    :param float f: smoothing factory
    :param int niter: maximum number of iterations
    :param float madclip: nMAD-clipping
    :return: estimated (smooth) values of *y*

    For more information, see:

    * William S. Cleveland: *Robust locally weighted regression and
      smoothing scatterplots*, Journal of the American Statistical
      Association, December 1979, volume 74, number 368, pp. 829-836.

    * William S. Cleveland and Susan J. Devlin: *Locally weighted
      regression: An approach to regression analysis by local
      fitting*, Journal of the American Statistical Association,
      September 1988, volume 83, number 403, pp. 596-610.

    See also: http://itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm

    Adapted from: Koders Code Search: lowess.py - Python

    Function from SNFactory ToolBox and wrote by Yannick Copin.
    """

    assert len(x)==len(y), "Incompatible input arrays"
    assert 0 < f < 1, "Smoothing parameter has to be in ]0,1["
    assert niter >= 0, "Nb of iterations has to be >= 0"
    assert madclip >= 0, "MAD-clipping has to be >= 0"

    n = len(x)
    r = int(np.ceil(f*n))
    h = np.array([ np.sort(np.abs(x-x[i]))[r] for i in range(n) ])
    # Tri-cube weighting function: (1-|x|**3)**3 for |x|<1, 0 otherwise
    w = ( 1 - np.clip(np.abs(x - x.reshape(-1,1))/h, 0., 1.)**3 )**3

    yest = np.zeros(n,'d')
    delta = np.ones(n,'d')
    for iteration in range(niter):      # Iterations
        for i in range(n):              # Weighted linear fit
            weights = delta * w[:,i]
            b = np.array([ (weights*y).sum(), (weights*y*x).sum() ])
            A = np.array([ [(weights).sum(),   (weights*x).sum()],
                          [(weights*x).sum(), (weights*x*x).sum()] ])
            beta = np.linalg.solve(A,b)
            yest[i] = beta[0] + beta[1]*x[i]
        # MAD-clipping
        residuals = y - yest
        mad = np.median(np.abs(residuals))
        delta = np.clip(residuals/(madclip*mad), -1, 1)
        delta = (1-delta**2)**2

    return yest


def correlation_CI(rho, n, cl=0.95):
    """Compute Pearson's correlation coefficient confidence interval,
    at (fractional) confidence level *cl*, for an observed correlation
    coefficient *rho* obtained on a sample of *n* (effective) points.

    cl=0.6827 corresponds to a 1-sigma error, 0.9973 for a 3-sigma
    error (`2*scipy.stats.norm.cdf(n)-1` or `1-2*sigma2pvalue(n)` for
    a n-sigma error).

    Sources: `Confidence Interval of rho
    <http://vassarstats.net/rho.html>`_, `Correlation CI
    <http://onlinestatbook.com/chapter8/correlation_ci.html>`_

    Function from SNFactory ToolBox and wrote by Yannick Copin.
    """

    assert -1 < rho < 1, "Correlation coefficient should be in ]-1,1["
    assert n >= 6, "Insufficient sample size"
    assert 0 < cl < 1, "Confidence level should be in ]0,1["

    z = np.arctanh(rho)                  # Fisher's transformation
    # z is normally distributed with std error = 1/sqrt(N-3)
    zsig = stats.distributions.norm.ppf(0.5*(cl+1)) / np.sqrt(n-3.)
    # Confidence interval on z is [z-zsig,z+zsig]

    return np.tanh([z-zsig,z+zsig])      # Confidence interval on rho

def pvalue2sigma(p):
    """Express the input one-sided *p*-value as a sigma equivalent
    significance from a normal distribution (the so-called *z*-value).

    =====  =======  =================
    sigma  p-value  terminology
    =====  =======  =================
    1      0.1587
    1.64   0.05     significant
    2      0.0228
    2.33   0.01     highly significant
    3      0.0013   evidence
    3.09   0.001
    5      2.9e-7   discovery
    =====  =======  =================

    >>> pvalue2sigma(1e-3) # p=0.1% corresponds to a ~3-sigma significance
    3.0902323061678132

    Function from SNFactory ToolBox and wrote by Yannick Copin.
    """
    return stats.distributions.norm.isf(p)  # isf = ppf(1 - p)

def correlation_significance(rho, n, directional=True, sigma=False):
    """Significance of (Pearson's or Spearman's) correlation
    coefficient *rho*, given the (effective) size *n* of the sample.

    If non-*directional*, this is the (two-sided) probability `p =
    Prob_N(|r| >= |rho|)` to find such an extreme correlation
    coefficient (no matter the sign of the correlation) from a purely
    uncorrelated population. If *directional*, this is the (one-sided)
    probability `p = Prob_N(r > rho)` for rho>0 (resp. `Prob_N(r <
    rho)` for rho<0). The directional (one-sided) probability is just
    half the non-directional (two-sided) one.

    If *sigma*, express the result as a sigma equivalent significance
    from a normal distribution (:func:`pvalue2sigma`).

    Sources: *Introduction to Error Analysis* (Taylor, 1997),
    `Significance of a Correlation Coefficient
    <http://vassarstats.net/rsig.html>`_
    
    Function from SNFactory ToolBox and wrote by Yannick Copin. 
    """

    assert -1 < rho < 1, "Correlation coefficient should be in ]-1,1["
    assert n >= 6, "Insufficient sample size"
    # t is distributed as Student's T distribution with DoF=n-2
    t = rho * np.sqrt((n - 2.)/(1 - rho**2))
    p = stats.distributions.t.sf(t, n-2)   # directional (one-sided) p-value
    if sigma:
        p = pvalue2sigma(p)
    elif not directional:               # non-directional (two-sided) p-value
        p *= 2
    return p


def neff_weighted(w, axis=None):
    """Compute the effective number of points from the weights:
    .. math::
       n_e = (\sum_i w_i)^2 / \sum_i w_i^2
       Function from SNFactory ToolBox and wrote by Yannick Copin.
    """
    aw = np.asarray(w)
    return aw.sum(axis=axis)**2/(aw**2).sum(axis=axis)


def correlation_weighted(x, y, w=None, axis=None,
                         error=False, confidence=0.6827, symmetric=False):
    """Compute (weighted) Pearson correlation coefficient between *x*
    and *y* along *axis*.

    **Weighting choice:** if *x* and *y* have (potentially correlated)
    errors, a logical choice is the inverse of the error ellipse
    area. For uncorrelated errors, this would correspond to :math:`w =
    1/(\sigma_x\sigma_y)`.

    **Error on weighted correlation:** once you have computed the
    (weighted) correlation, use :func:`correlation_CI` (resp.
    :func:`correlation_significance`) to compute associated confidence
    interval (resp. significance). You should then use an *effective*
    number of weighted points (see :func:`neff_weighted`).

    Source: `Weighted correlation
    <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Calculating_a_weighted_correlation>`_

    Function from SNFactory ToolBox and wrote by Yannick Copin.
    """
    #from ToolBox.Arrays import unsqueeze

    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape==y.shape, "Incompatible data arrays x and y"

    # Weights
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.where(np.isfinite(w), w, 0) # Discard NaN's and Inf's
        assert w.shape==x.shape, "Weight array w incompatible with data arrays"

    # Weighted means
    mx = np.average(x, weights=w, axis=axis)
    my = np.average(y, weights=w, axis=axis)
    # Residuals around weighted means
    xm = x - np.expand_dims(mx, axis)
    ym = y - np.expand_dims(my, axis)
    # Weighted covariance
    cov,sumw = np.average(xm*ym, weights=w, axis=axis, returned=True)
    # Weighted variances
    vx = np.average(xm**2, weights=w, axis=axis)
    vy = np.average(ym**2, weights=w, axis=axis)

    # Weighted correlation
    rho = cov/np.sqrt(vx*vy)
    
    if not error:
        return rho

    if axis is not None:
        raise NotImplementedError("Weighted correlation confidence interval "
                                  "not implemented for nD-arrays.")

    # Compute error on correlation coefficient using effective number of points
    rho_dn,rho_up = correlation_CI(rho, n=neff_weighted(w), cl=confidence)
    drm = rho-rho_dn
    drp = rho_up-rho

    if symmetric:
        return rho,np.hypot(drm,drp)/1.4142135623730951 # Symmetrized error
    else:
        return rho,drm,drp                             # Assymmetric errors


def flbda2fnu(x, y, var=None, backward=False):
    """Convert *x* [A], *y* [erg/s/cm2/A] to *y* [erg/s/cm2/Hz]. 
    Set `var=var(y)` to get variance. Function from SNFactory ToolBox and wrote by Yannick Copin."""

    f = x**2 / 299792458. * 1.e-10 # Conversion factor

    if backward:
        f = 1./f
    if var is None:                # Return converted signal
        return y * f
    else:                          # Return converted variance
        return var * f**2

def flbda2ABmag(x, y, ABmag0=48.59, var=None):
    """Convert *x* [A], *y* [erg/s/cm2/A] to `ABmag =
    -2.5*log10(erg/s/cm2/Hz) - ABmag0`. Set `var=var(y)` to get
    variance.  Function from SNFactory ToolBox and wrote by Yannick Copin."""

    z = flbda2fnu(x,y)
    if var is None:
        return -2.5*np.log10(z) - ABmag0
    else:
        return (2.5/np.log(10)/z)**2 * flbda2fnu(x,y,var=var)
