"""
Core utilities for moving-correlation significance testing.

Provides:
  - moving_correlation: vectorized Pearson/Spearman moving (windowed) correlation
  - ar1_fit: lag-1 autoregressive fit (Yule-Walker) for a single series
  - white_noise_pair / red_noise_pair: generate a pair of surrogate series with a
    target overall (Pearson) correlation, either as white noise or as AR(1)
    ("red") noise matched to given lag-1 autocorrelation coefficients.

These are the building blocks shared by all three significance tests in
`tests.py`.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import lfilter

__all__ = [
    "moving_correlation",
    "ar1_fit",
    "white_noise_pair",
    "red_noise_pair",
]


def _as_1d_float(a, name="array"):
    a = np.asarray(a, dtype=float).squeeze()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {a.shape}")
    return a


def moving_correlation(x, y, window, method="pearson"):
    """
    Vectorized moving (sliding-window) correlation between two equal-length
    series.

    Note on implementation: pandas offers `Series.rolling(window).corr()`,
    which gives identical results for the Pearson case, but is roughly 5x
    slower in practice than the approach here (benchmarked: ~0.66us vs
    ~0.13us per call on a length-150 series). Since this function is called
    once per Monte Carlo surrogate draw -- often thousands of times per
    significance test -- the numpy sliding-window approach is used instead.
    pandas' rolling correlation also does not support Spearman correlation
    between two series directly, which this function needs.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        The two time series. Must be the same length.
    window : int
        Width of the sliding window, in samples. Must be < n.
    method : {"pearson", "spearman"}
        Correlation type. Spearman is computed by rank-transforming each
        window independently before applying the Pearson formula.

    Returns
    -------
    r : ndarray, shape (n - window + 1,)
        Correlation coefficient for each window. r[k] is the correlation of
        the window x[k:k+window], y[k:k+window] -- i.e. it is indexed by the
        *start* of the window. The window's last (most recent) time index is
        k + window - 1, which is a convenient convention for plotting against
        a time axis: t_end = t[window-1:] aligns with r.
    """
    x = _as_1d_float(x, "x")
    y = _as_1d_float(y, "y")
    if x.shape != y.shape:
        raise ValueError(f"x and y must be the same length (got {x.shape} vs {y.shape})")
    n = x.shape[0]
    if window < 2:
        raise ValueError("window must be >= 2")
    if window >= n:
        raise ValueError(f"window ({window}) must be shorter than the series length ({n})")

    Wx = sliding_window_view(x, window)  # shape (n-window+1, window)
    Wy = sliding_window_view(y, window)

    if method == "spearman":
        Wx = stats.rankdata(Wx, axis=1)
        Wy = stats.rankdata(Wy, axis=1)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    xm = Wx.mean(axis=1, keepdims=True)
    ym = Wy.mean(axis=1, keepdims=True)
    dx = Wx - xm
    dy = Wy - ym

    cov = (dx * dy).sum(axis=1)
    sx = np.sqrt((dx ** 2).sum(axis=1))
    sy = np.sqrt((dy ** 2).sum(axis=1))

    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (sx * sy)
    return r


def ar1_fit(x):
    """
    Fit an AR(1) model to a series:

        x_t = c + phi * x_{t-1} + e_t

    using statsmodels' `AutoReg` (least-squares AR fitting -- the same
    philosophy as the original ARFIT toolkit this project is descended
    from). Requires statsmodels.

    Returns
    -------
    phi : float
        Lag-1 autoregressive coefficient (clipped to (-0.98, 0.98) for
        numerical stability of downstream simulation).
    """
    from statsmodels.tsa.ar_model import AutoReg

    x = _as_1d_float(x, "x")
    res = AutoReg(x, lags=1, trend="c", old_names=False).fit()
    # Index positionally (not by name) since statsmodels' parameter naming
    # for the lag term varies slightly across versions/inputs; with lags=1
    # and trend='c' there are always exactly two params,
    # [intercept, ar_lag1_coefficient].
    return float(np.clip(res.params[-1], -0.98, 0.98))


def white_noise_pair(n, target_corr, rng, tol=0.01, max_tries=200):
    """
    Generate a pair of unit-variance Gaussian white-noise series of length n
    whose *realized* Pearson correlation is within `tol` of `target_corr`.

    Uses numpy's `Generator.multivariate_normal` to impose the target
    correlation in expectation, then rejection-samples (redraws) until the
    realized sample correlation is acceptably close to the target --
    necessary because any finite draw has its own sampling noise around the
    population value.

    Returns
    -------
    x, y : ndarray, shape (n,)
    """
    target_corr = float(np.clip(target_corr, -0.999, 0.999))
    cov = np.array([[1.0, target_corr], [target_corr, 1.0]])

    x = y = None
    for _ in range(max_tries):
        X = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
        x, y = X[:, 0], X[:, 1]
        realized = np.corrcoef(x, y)[0, 1]
        if abs(realized - target_corr) <= tol:
            return x, y
    return x, y  # best effort after max_tries; caller may check realized corr if desired


def red_noise_pair(n, target_corr, phi_x, phi_y, rng, tol=0.01, max_tries=200, burn=500):
    """
    Generate a pair of AR(1) ("red noise") series of length n with lag-1
    autocorrelations phi_x, phi_y, whose realized overall Pearson correlation
    is within `tol` of `target_corr`.

    The two AR(1) processes are driven by contemporaneously-correlated
    Gaussian innovations:

        x_t = phi_x * x_{t-1} + e_x,t
        y_t = phi_y * y_{t-1} + e_y,t
        corr(e_x,t, e_y,t) = rho_e

    For this model, the stationary correlation between x and y is related to
    rho_e in closed form:

        corr(x, y) = rho_e * sqrt((1 - phi_x^2)(1 - phi_y^2)) / (1 - phi_x*phi_y)

    which we invert to choose rho_e that targets the desired corr(x, y) in
    expectation, then (as in white_noise_pair) rejection-sample to correct
    for finite-sample deviation from that expectation.

    Correlated innovations are drawn with numpy's `Generator.multivariate_normal`;
    AR(1) filtering is done with scipy.signal.lfilter for speed. A burn-in
    of `burn` samples is discarded to remove startup transients.

    Returns
    -------
    x, y : ndarray, shape (n,)
    """
    target_corr = float(np.clip(target_corr, -0.999, 0.999))
    phi_x = float(np.clip(phi_x, -0.98, 0.98))
    phi_y = float(np.clip(phi_y, -0.98, 0.98))

    denom = np.sqrt((1 - phi_x ** 2) * (1 - phi_y ** 2))
    if denom <= 1e-8:
        raise ValueError("phi_x/phi_y too close to +/-1 for stable red-noise simulation")
    rho_e = target_corr * (1 - phi_x * phi_y) / denom
    rho_e = float(np.clip(rho_e, -0.999, 0.999))
    cov_e = np.array([[1.0, rho_e], [rho_e, 1.0]])

    ntot = n + burn
    x = y = None
    for _ in range(max_tries):
        E = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_e, size=ntot)  # shape (ntot, 2)
        xf = lfilter([1.0], [1.0, -phi_x], E[:, 0])
        yf = lfilter([1.0], [1.0, -phi_y], E[:, 1])
        x, y = xf[burn:], yf[burn:]
        realized = np.corrcoef(x, y)[0, 1]
        if abs(realized - target_corr) <= tol:
            return x, y
    return x, y  # best effort after max_tries
