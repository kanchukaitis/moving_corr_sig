"""
Three significance tests for moving (windowed) correlation analyses, all
built on a shared Monte Carlo engine (see `_mc_ensemble`).

  1. std_test        -- Gershunov et al. (2001): is the running-correlation
                         trace more (or less) variable through time than
                         sampling noise alone predicts, given the overall
                         (background) correlation between the two series?

  2. peak_test        -- Is the single highest windowed correlation higher
                          than chance predicts? Two conditioning modes:
                            condition="zero"     -> null = unrelated series
                                                     (existence / detection test)
                            condition="observed" -> null = series related at
                                                     the observed overall
                                                     correlation (elevation-
                                                     above-baseline test)

  3. range_test       -- Is the swing between the highest and lowest windowed
                          correlation (max - min) bigger than the swing
                          sampling noise alone would produce around the
                          overall (background) correlation? A more targeted
                          alternative/complement to std_test for the
                          "are the highs and lows really different" question.

All three share one Monte Carlo surrogate-generation engine so results are
directly comparable, and each can be run with white-noise or red-noise
(AR(1)) surrogates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from .core import moving_correlation, ar1_fit, white_noise_pair, red_noise_pair
except ImportError:
    # Falls back to a plain sibling-module import when tests.py is loaded
    # standalone (not as part of the moving_corr_sig package) -- e.g. if
    # core.py and tests.py were downloaded as loose files rather than as
    # a moving_corr_sig/ folder.
    from core import moving_correlation, ar1_fit, white_noise_pair, red_noise_pair

__all__ = [
    "TestResult",
    "std_test",
    "peak_test",
    "range_test",
    "MovingCorrelationTest",
]


@dataclass
class TestResult:
    """Container for the result of a single significance test."""
    name: str
    observed: float
    null_distribution: np.ndarray
    p_value: float
    p_value_lower: Optional[float] = None  # only populated for two-tailed tests (std_test)
    quantiles: dict = field(default_factory=dict)
    noise: str = "white"
    target_corr: float = 0.0
    iters: int = 0
    window: int = 0
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        base = (f"TestResult(name={self.name!r}, observed={self.observed:.4f}, "
                f"p_value={self.p_value:.4f}")
        if self.p_value_lower is not None:
            base += f", p_value_lower={self.p_value_lower:.4f}"
        base += f", noise={self.noise!r}, target_corr={self.target_corr:.3f}, iters={self.iters})"
        return base

    def summary(self):
        lines = [f"{self.name}"]
        lines.append(f"  noise model      : {self.noise}")
        lines.append(f"  target corr (c)  : {self.target_corr:.3f}")
        lines.append(f"  window           : {self.window}")
        lines.append(f"  Monte Carlo iters: {self.iters}")
        lines.append(f"  observed value   : {self.observed:.4f}")
        for k, v in sorted(self.quantiles.items()):
            lines.append(f"  null {k:>6s} pct : {v:.4f}")
        lines.append(f"  p-value (upper)  : {self.p_value:.4f}")
        if self.p_value_lower is not None:
            lines.append(f"  p-value (lower)  : {self.p_value_lower:.4f}")
        return "\n".join(lines)

    def threshold(self, level):
        """
        Return the null-distribution quantile at an arbitrary confidence
        level (e.g. 0.90, 0.95, 0.975, 0.99), computed on demand from the
        stored `null_distribution` -- not limited to whichever levels were
        precomputed into `quantiles` at test-run time.

        For `peak_test` results in particular, this is the significance
        threshold you'd draw as a horizontal line alongside the observed
        moving-correlation trace: since the null distribution is built from
        the *maximum* correlation seen anywhere in each surrogate trace, a
        single threshold value already accounts for having searched the
        whole record, and applies uniformly across the plot -- points on the
        observed trace above this line are the significant peaks.
        """
        return float(np.quantile(self.null_distribution, level))


def _mc_ensemble(n, window, target_corr, iters, noise, method, rng,
                  phi_x=None, phi_y=None, tol=0.01, max_tries=200):
    """
    Run `iters` Monte Carlo draws of surrogate pairs (white or red noise) at
    the given target overall correlation, compute the moving-correlation
    trace for each, and return arrays of summary statistics (std, max, min,
    range) across draws -- one shared ensemble that all three tests draw from.
    """
    if noise not in ("white", "red"):
        raise ValueError("noise must be 'white' or 'red'")
    if noise == "red" and (phi_x is None or phi_y is None):
        raise ValueError("phi_x and phi_y are required when noise='red'")

    stds = np.empty(iters)
    maxs = np.empty(iters)
    mins = np.empty(iters)

    for i in range(iters):
        if noise == "white":
            sx, sy = white_noise_pair(n, target_corr, rng, tol=tol, max_tries=max_tries)
        else:
            sx, sy = red_noise_pair(n, target_corr, phi_x, phi_y, rng, tol=tol, max_tries=max_tries)
        r = moving_correlation(sx, sy, window, method=method)
        stds[i] = np.nanstd(r)
        maxs[i] = np.nanmax(r)
        mins[i] = np.nanmin(r)

    return {"std": stds, "max": maxs, "min": mins, "range": maxs - mins}


def _resolve_ar1(x, y, noise):
    if noise == "red":
        phi_x, _ = ar1_fit(x)
        phi_y, _ = ar1_fit(y)
        return phi_x, phi_y
    return None, None


def std_test(x, y, window, iters=1000, noise="white", method="pearson",
             target_corr=None, rng=None, tol=0.01, max_tries=200,
             levels=(0.01, 0.05, 0.10, 0.90, 0.95, 0.99)):
    """
    Gershunov et al. (2001) test: is the standard deviation of the observed
    moving-correlation trace larger (or smaller) than expected from sampling
    variability alone, given the overall correlation between x and y?

    Two-tailed: p_value is the upper-tail p (trace MORE variable than
    chance); p_value_lower is the lower-tail p (trace MORE stable than
    chance, as Gershunov et al. found for ENSO-AIR).

    `levels` controls which quantiles of the null distribution are
    precomputed into the returned TestResult's `.quantiles` dict (e.g. for
    printing in `.summary()`). Any level can still be queried afterward via
    `.threshold(level)`, precomputed or not.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    r_obs = moving_correlation(x, y, window, method=method)
    std_obs = float(np.nanstd(r_obs))

    if target_corr is None:
        target_corr = float(np.corrcoef(x, y)[0, 1])

    phi_x, phi_y = _resolve_ar1(x, y, noise)
    ens = _mc_ensemble(n, window, target_corr, iters, noise, method, rng,
                        phi_x, phi_y, tol, max_tries)
    null = ens["std"]

    p_upper = float(np.mean(null >= std_obs))
    p_lower = float(np.mean(null <= std_obs))
    quantiles = {f"{int(round(q*1000))/10:g}": float(np.quantile(null, q)) for q in levels}

    return TestResult(
        name="Gershunov std test",
        observed=std_obs,
        null_distribution=null,
        p_value=p_upper,
        p_value_lower=p_lower,
        quantiles=quantiles,
        noise=noise,
        target_corr=target_corr,
        iters=iters,
        window=window,
    )


def peak_test(x, y, window, iters=1000, noise="white", condition="observed",
              method="pearson", rng=None, tol=0.01, max_tries=200,
              levels=(0.90, 0.95, 0.99)):
    """
    Is the single highest windowed correlation between x and y higher than
    chance predicts?

    condition="zero"     : null = x and y are entirely unrelated
                            (existence / detection test -- "was there ever
                            any real relationship, anywhere in the record?")
    condition="observed" : null = x and y are related at their observed
                            overall correlation (elevation test -- "is this
                            peak notably higher than the established
                            background relationship, or just sampling wobble
                            around it?")

    One-tailed upper test.

    `levels` controls which quantiles of the null distribution are
    precomputed into the returned TestResult's `.quantiles` dict -- these
    are the significance threshold(s) you'd plot as horizontal line(s)
    alongside the observed moving-correlation trace (see
    `MovingCorrelationTest.plot_peak_test` for a ready-made version of that
    plot). Any level can still be queried afterward via `.threshold(level)`,
    precomputed or not.
    """
    if condition not in ("zero", "observed"):
        raise ValueError("condition must be 'zero' or 'observed'")
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    r_obs = moving_correlation(x, y, window, method=method)
    max_obs = float(np.nanmax(r_obs))

    target_corr = 0.0 if condition == "zero" else float(np.corrcoef(x, y)[0, 1])

    phi_x, phi_y = _resolve_ar1(x, y, noise)
    ens = _mc_ensemble(n, window, target_corr, iters, noise, method, rng,
                        phi_x, phi_y, tol, max_tries)
    null = ens["max"]

    p = float(np.mean(null >= max_obs))
    quantiles = {f"{int(round(q*1000))/10:g}": float(np.quantile(null, q)) for q in levels}

    return TestResult(
        name=f"Peak correlation test (condition={condition})",
        observed=max_obs,
        null_distribution=null,
        p_value=p,
        quantiles=quantiles,
        noise=noise,
        target_corr=target_corr,
        iters=iters,
        window=window,
        extra={"condition": condition},
    )


def range_test(x, y, window, iters=1000, noise="white", method="pearson",
               target_corr=None, rng=None, tol=0.01, max_tries=200,
               levels=(0.90, 0.95, 0.99)):
    """
    Is the swing between the highest and lowest windowed correlation
    (max - min) bigger than sampling noise around the overall (background)
    correlation predicts? A targeted alternative to std_test for asking
    whether the "highs" and "lows" of the trace are meaningfully different
    from one another, rather than testing overall dispersion.

    One-tailed upper test. Always conditions on the observed (or supplied)
    overall correlation -- a zero-correlation version of this test would
    conflate "is there a relationship" with "does it swing," which is why
    peak_test (not range_test) offers the zero-correlation option.

    `levels` controls which quantiles of the null distribution are
    precomputed into the returned TestResult's `.quantiles` dict. Any level
    can still be queried afterward via `.threshold(level)`, precomputed or not.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    r_obs = moving_correlation(x, y, window, method=method)
    range_obs = float(np.nanmax(r_obs) - np.nanmin(r_obs))

    if target_corr is None:
        target_corr = float(np.corrcoef(x, y)[0, 1])

    phi_x, phi_y = _resolve_ar1(x, y, noise)
    ens = _mc_ensemble(n, window, target_corr, iters, noise, method, rng,
                        phi_x, phi_y, tol, max_tries)
    null = ens["range"]

    p = float(np.mean(null >= range_obs))
    quantiles = {f"{int(round(q*1000))/10:g}": float(np.quantile(null, q)) for q in levels}

    return TestResult(
        name="High-low range test",
        observed=range_obs,
        null_distribution=null,
        p_value=p,
        quantiles=quantiles,
        noise=noise,
        target_corr=target_corr,
        iters=iters,
        window=window,
    )


class MovingCorrelationTest:
    """
    Convenience wrapper: pass in two real time series once, then run any of
    the three significance tests against them without re-specifying the
    data, window, or method each time.

    Example
    -------
    >>> mct = MovingCorrelationTest(seriesA, seriesB, window=15, noise="red")
    >>> mct.r                      # the observed moving-correlation trace
    >>> mct.std_test()             # Gershunov-style variability test
    >>> mct.peak_test()            # peak/elevation test
    >>> mct.range_test()           # high-vs-low swing test
    """

    def __init__(self, x, y, window, method="pearson", noise="white", rng=None):
        x = np.asarray(x, dtype=float).squeeze()
        y = np.asarray(y, dtype=float).squeeze()
        if x.shape != y.shape:
            raise ValueError(f"x and y must be the same length (got {x.shape} vs {y.shape})")
        self.x = x
        self.y = y
        self.window = window
        self.method = method
        self.noise = noise  # default noise model; can be overridden per-call
        self.rng = rng or np.random.default_rng()

        self.r = moving_correlation(x, y, window, method=method)
        self.overall_corr = float(np.corrcoef(x, y)[0, 1])
        if noise == "red":
            self.phi_x, _ = ar1_fit(x)
            self.phi_y, _ = ar1_fit(y)
        else:
            self.phi_x = self.phi_y = None

    def std_test(self, iters=1000, noise=None, target_corr=None, **kwargs):
        return std_test(self.x, self.y, self.window, iters=iters,
                         noise=noise or self.noise, method=self.method,
                         target_corr=target_corr, rng=self.rng, **kwargs)

    def peak_test(self, iters=1000, noise=None, condition="observed", **kwargs):
        return peak_test(self.x, self.y, self.window, iters=iters,
                          noise=noise or self.noise, condition=condition,
                          method=self.method, rng=self.rng, **kwargs)

    def range_test(self, iters=1000, noise=None, target_corr=None, **kwargs):
        return range_test(self.x, self.y, self.window, iters=iters,
                           noise=noise or self.noise, method=self.method,
                           target_corr=target_corr, rng=self.rng, **kwargs)

    def run_all(self, iters=1000, noise=None, peak_condition="observed", **kwargs):
        """Run all three tests and return them as a dict keyed by test name."""
        return {
            "std_test": self.std_test(iters=iters, noise=noise, **kwargs),
            "peak_test": self.peak_test(iters=iters, noise=noise, condition=peak_condition, **kwargs),
            "range_test": self.range_test(iters=iters, noise=noise, **kwargs),
        }

    @property
    def time_index(self):
        """
        Index positions (into the original x/y arrays) that `self.r` aligns
        to -- the *end* of each window, i.e. `window - 1, window, ..., n - 1`.
        Convenient as an x-axis when plotting `self.r` against a real time
        vector: `real_time[mct.time_index]`.
        """
        return np.arange(len(self.r)) + self.window - 1

    def plot_peak_test(self, iters=1000, noise=None, condition="observed",
                        levels=(0.90, 0.95), t=None, ax=None,
                        highlight_exceedances=True, peak_result=None):
        """
        Plot the observed moving-correlation trace with horizontal
        significance threshold line(s) from `peak_test` overlaid -- e.g. for
        a trace that spends most of its time below the null model but
        occasionally rises above it, this draws the threshold(s) so the
        significant excursions are visible at a glance.

        Parameters
        ----------
        iters, noise, condition : passed to `peak_test` (ignored if
            `peak_result` is supplied directly).
        levels : sequence of floats
            Confidence levels to draw as threshold lines, e.g. (0.90, 0.95).
        t : array-like, optional
            Time axis to plot against, same length as the original x/y. If
            omitted, uses `self.time_index` (window-end sample positions).
        ax : matplotlib Axes, optional
            Axes to draw into; a new figure/axes is created if omitted.
        highlight_exceedances : bool
            If True, mark points on the trace that exceed the highest
            requested threshold level.
        peak_result : TestResult, optional
            Reuse an already-computed `peak_test` result (e.g. to avoid
            rerunning the Monte Carlo simulation) instead of running a new one.

        Returns
        -------
        ax : matplotlib Axes
        result : TestResult
            The peak_test result used to draw the threshold line(s).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot_peak_test requires matplotlib. Install it with "
                "`pip install matplotlib` or `pip install moving_corr_sig[notebook]`."
            ) from e

        if peak_result is None:
            peak_result = self.peak_test(iters=iters, noise=noise, condition=condition,
                                          levels=levels)

        t_axis = np.asarray(t)[self.time_index] if t is not None else self.time_index

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(t_axis, self.r, color="k", lw=1.2, label="moving correlation", zorder=3)

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
        max_thresh = None
        for i, lvl in enumerate(sorted(levels)):
            thresh = peak_result.threshold(lvl)
            max_thresh = thresh if max_thresh is None else max(max_thresh, thresh)
            ax.axhline(thresh, ls="--", color=colors[i % len(colors)],
                       label=f"{lvl*100:g}% significance threshold", zorder=2)

        if highlight_exceedances and max_thresh is not None:
            exceed = self.r > max_thresh
            if np.any(exceed):
                ax.scatter(t_axis[exceed], self.r[exceed], color="crimson", zorder=4,
                           label=f"exceeds {max(levels)*100:g}% threshold", s=20)

        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("moving correlation")
        ax.legend(loc="best", fontsize="small")
        return ax, peak_result
