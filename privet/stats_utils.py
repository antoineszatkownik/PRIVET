import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import betainc

###########################################################################
# Least-squares fits on the linearised empirical CDF of the N sorted
# train-train 1-NN distances, on the window [start_idx, end_idx).
# The intercept is log(N A).
###########################################################################

def fit_nearest_neighbor_cdf_weibull(z_star, start_idx, end_idx):
    """
    Fit P(z* < x) = 1 - exp(-N A x^alpha) by least squares on
    Y = ln(-ln(survival)) = ln(N A) + alpha ln(x).
    Returns intercept = ln(N A), alpha, their standard errors, and the residual MSE.
    """
    z_sorted = np.sort(z_star)
    N = len(z_sorted)
    survival = 1 - (np.arange(N) + 1) / N

    Y = np.log(-np.log(survival[start_idx:end_idx]))
    X = np.log(z_sorted[start_idx:end_idx])

    X_design = np.vstack([np.ones_like(X), X]).T
    beta, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    intercept, alpha = beta[0], beta[1]

    residuals = Y - X_design @ beta
    mse = np.sum(residuals**2) / (len(Y) - 2)
    cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)
    std_err_intercept, std_err_alpha = np.sqrt(np.diag(cov_matrix))
    return intercept, alpha, std_err_intercept, std_err_alpha, mse


def fit_nearest_neighbor_cdf_gumbel(z_star, start_idx, end_idx):
    """
    Fit P(z* < x) = 1 - exp(-N A exp(B x)) by least squares on
    Y = ln(-ln(survival)) = ln(N A) + B x.
    Returns intercept = ln(N A), B, their standard errors, and the residual MSE.
    """
    z_sorted = np.sort(z_star)
    N = len(z_sorted)
    survival = 1 - (np.arange(N) + 1) / N

    Y = np.log(-np.log(survival[start_idx:end_idx]))
    X = z_sorted[start_idx:end_idx]

    X_design = np.vstack([np.ones_like(X), X]).T
    beta, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    intercept, B = beta[0], beta[1]

    residuals = Y - X_design @ beta
    mse = np.sum(residuals**2) / (len(Y) - 2)
    cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)
    std_err_intercept, std_err_B = np.sqrt(np.diag(cov_matrix))
    return intercept, B, std_err_intercept, std_err_B, mse


###########################################################################
# Maximum-likelihood fits, same model, written with Lambda(x) = exp(c) x^alpha
# (Weibull) or exp(c + B x) (Gumbel), c = ln(N A).
# Censored likelihood: the points inside the window contribute their density
# lambda(x) exp(-Lambda(x)) with lambda = dLambda/dx; the start_idx points below
# the window contribute P(z* <= a) = 1 - exp(-Lambda(a)); the N - end_idx points
# above it contribute P(z* >= b) = exp(-Lambda(b)); a, b = window ends.
# The counts outside the window pin c: a likelihood restricted to the window
# alone barely depends on c in the lower tail, where exp(-Lambda) ~ 1.
###########################################################################

def _censored_mle(z_star, start_idx, end_idx, intercept0, slope0, family):
    z_sorted = np.sort(np.asarray(z_star, dtype=float))
    N = len(z_sorted)
    x = z_sorted[start_idx:end_idx]
    a, b = x[0], x[-1]
    n_below, n_above = start_idx, N - end_idx
    if family == "Weibull":       # Lambda = exp(c + alpha ln x)
        tx, ta, tb = np.log(x), np.log(a), np.log(b)
    else:                         # Lambda = exp(c + B x)
        tx, ta, tb = x, a, b
    # Optimise in (c + s t0, s) with t0 the mean of t over the window: c and s are
    # strongly correlated otherwise, and the optimiser stops early.
    t0 = tx.mean()
    tx, ta, tb = tx - t0, ta - t0, tb - t0

    def neg_log_likelihood_and_gradient(theta):
        c, s = theta              # s = alpha or B; here c stands for c + s t0
        L, La, Lb = np.exp(c + s * tx), np.exp(c + s * ta), np.exp(c + s * tb)
        # ln lambda(x) = ln s + c + s t(x) + ln(dt/dx); the last term does not depend on theta
        ll = (np.sum(np.log(s) + c + s * tx - L)
              + n_below * np.log(-np.expm1(-La))
              - n_above * Lb)
        w_a = La / np.expm1(La)   # d ln(1 - exp(-Lambda(a))) / d ln Lambda(a)
        d_c = np.sum(1 - L) + n_below * w_a - n_above * Lb
        d_s = np.sum(1 / s + tx - L * tx) + n_below * w_a * ta - n_above * Lb * tb
        if not np.isfinite(ll):
            return np.inf, np.zeros(2)
        return -ll, -np.array([d_c, d_s])

    res = minimize(neg_log_likelihood_and_gradient, x0=[intercept0 + slope0 * t0, slope0],
                   jac=True, method="L-BFGS-B", bounds=[(None, None), (1e-12, None)],
                   options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
    c_centred, slope = res.x
    return c_centred - slope * t0, slope, res.fun / N


def mle_fit_nearest_neighbor_cdf_weibull(z_star, start_idx, end_idx, intercept0, alpha0):
    """Returns intercept = ln(N A), alpha, mean negative log-likelihood."""
    return _censored_mle(z_star, start_idx, end_idx, intercept0, alpha0, "Weibull")


def mle_fit_nearest_neighbor_cdf_gumbel(z_star, start_idx, end_idx, intercept0, B0):
    """Returns intercept = ln(N A), B, mean negative log-likelihood."""
    return _censored_mle(z_star, start_idx, end_idx, intercept0, B0, "Gumbel")


###########################################################################
# The fitted law
###########################################################################

class EVDLaw:
    """
    Law of the 1-NN distance u to a reference set of n points (paper Eq. 1):
        \hat G_n(u) = 1 - exp(-Lambda_n(u)),
        Lambda_n(u) = n A u^alpha       Weibull
        Lambda_n(u) = n A exp(B u)      Gumbel
    A 1-NN distance is the minimum over n independent points, so Lambda is
    proportional to n, and one law serves reference sets of any size.

    upper_tail="empirical": above the fit window, Lambda is read off the empirical
    CDF F of the distances the law was fitted on, Lambda_n(u) = -(n / n_fit) ln(1 - F(u)),
    instead of extrapolating the fit.
    """

    def __init__(self, family, log_A, param2, d_sorted, n_fit, start_idx, end_idx,
                 fit_method, criterion):
        self.family = family            # "Weibull" or "Gumbel"
        self.log_A = log_A              # ln A, A = amplitude per reference point
        self.param2 = param2            # alpha (Weibull) or B (Gumbel)
        self.d_sorted = d_sorted        # distances the law was fitted on
        self.n_fit = n_fit              # reference size of those distances
        self.window = (d_sorted[start_idx], d_sorted[end_idx - 1])   # fit window, in distance
        self.fit_method = fit_method
        self.criterion = criterion      # MSE (ols) or mean negative log-likelihood (mle), both families

    def log_hazard(self, u, n_ref, upper_tail="extrapolate"):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        with np.errstate(divide="ignore", over="ignore"):
            # computed in log space: for large alpha, A underflows and u^alpha overflows
            if self.family == "Weibull":
                log_L = np.log(n_ref) + self.log_A + self.param2 * np.log(u)
            else:
                log_L = np.log(n_ref) + self.log_A + self.param2 * u
            if upper_tail == "empirical":
                above = u > self.window[1]
                F = np.searchsorted(self.d_sorted, u[above], side="right") / len(self.d_sorted)
                log_L[above] = np.log(n_ref / self.n_fit) + np.log(-np.log1p(-F))
        return log_L

    def cdf(self, u, n_ref, upper_tail="extrapolate"):
        """\hat G_{n_ref}(u) = 1 - exp(-Lambda), via -expm1: no rounding to 0 for small Lambda."""
        return -np.expm1(-np.exp(self.log_hazard(u, n_ref, upper_tail)))


def fit_evd(d_sorted, partition_fit_start, partition_fit_end, fit_method="ols", family=None,
            loo_correction=False):
    """
    Fit Weibull and Gumbel laws on the window [start, end) (fractions) of the sorted
    1-NN distances d_sorted, and keep `family` if given, else the one with the lower
    MSE (ols) or negative log-likelihood (mle).
    loo_correction: the distances are minima over the n - 1 other points of their own
    set, so the fit estimates (n - 1) A rather than n A.
    """
    N = len(d_sorted)
    start_idx = int(np.ceil(partition_fit_start * N))
    end_idx = int(partition_fit_end * N)
    if end_idx - start_idx < 3:
        raise ValueError("fit window holds fewer than 3 distances; widen it")
    if d_sorted[start_idx] <= 0:
        raise ValueError("fit window starts at a zero distance (duplicated samples); "
                         "start it higher")
    if fit_method not in ("ols", "mle"):
        raise ValueError("fit_method must be 'ols' or 'mle'")

    fits = {}
    for fam, ols, mle in (
            ("Weibull", fit_nearest_neighbor_cdf_weibull, mle_fit_nearest_neighbor_cdf_weibull),
            ("Gumbel", fit_nearest_neighbor_cdf_gumbel, mle_fit_nearest_neighbor_cdf_gumbel)):
        intercept, slope, _, _, criterion = ols(d_sorted, start_idx, end_idx)
        if fit_method == "mle":     # least squares as starting point
            intercept, slope, criterion = mle(d_sorted, start_idx, end_idx, intercept, slope)
        fits[fam] = (intercept, slope, criterion)

    if family is None:
        family = "Gumbel" if fits["Gumbel"][2] < fits["Weibull"][2] else "Weibull"
    intercept, slope, _ = fits[family]
    n_fit = N - 1 if loo_correction else N
    return EVDLaw(family, intercept - np.log(n_fit),      # the fitted intercept is ln(n_fit A)
                  slope, d_sorted, n_fit, start_idx, end_idx, fit_method,
                  {fam: f[2] for fam, f in fits.items()})


###########################################################################
# Binomial survival function in log10, robust far into the tail.
# pi_r = P(Binom(M, G) >= r) = safe_log10_sf(r - 1, M, G) in log10.
###########################################################################

LN10 = np.log(10.0)


@njit
def _manual_logsf_natural_numba(k, n, p):
    """ln P(X > k) for X ~ Binom(n, p), summing j = n ... k+1 in log space."""
    log_p = np.log(p)
    log_q = np.log1p(-p)
    lj = n * log_p                # ln P(X = n)
    total = lj
    for j in range(n - 1, k, -1):
        lj = lj + np.log(j + 1) - np.log(n - j) + log_q - log_p
        M = total if total > lj else lj
        total = M + np.log(np.exp(total - M) + np.exp(lj - M))
    return total


def safe_log10_sf(k, n, p):
    """log10 P(X > k) for X ~ Binom(n, p)."""
    if p == 0.0:
        return -np.inf
    if p == 1.0:
        return 0.0 if k < n else -np.inf
    with np.errstate(divide="ignore"):
        nat = np.log(betainc(k + 1, n - k, p))       # incomplete beta, in C
    if np.isfinite(nat):
        return nat / LN10
    return _manual_logsf_natural_numba(k, n, p) / LN10   # underflow fallback


log10_sf = np.vectorize(safe_log10_sf, otypes=[float])   # elementwise over arrays
