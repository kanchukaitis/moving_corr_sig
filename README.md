# moving_corr_sig

Significance tests for moving (windowed) correlation analyses between two
time series, built on a shared Monte Carlo surrogate-generation engine with
switchable white-noise / red-noise (AR(1)) nulls.

Three complementary tests:

| Test | Question | Null conditions on |
|---|---|---|
| `std_test` | Is the running-correlation trace *more* (or *less*) variable through time than sampling noise alone predicts? (Gershunov et al. 2001) | Overall (background) correlation `c` |
| `peak_test` | Is the single highest windowed correlation higher than chance predicts? | `c = 0` (existence) **or** `c = observed` (elevation above baseline) |
| `range_test` | Is the swing between the highest and lowest windowed correlation bigger than chance predicts? | Overall (background) correlation `c` |

## Installation

```bash
pip install git+https://github.com/kanchukaitis/moving_corr_sig.git
```

Pin to a specific tag or commit for reproducibility:

```bash
pip install git+https://github.com/kanchukaitis/moving_corr_sig.git@v0.1.0
```

Optional extra:

```bash
# for running the demonstration notebook / plot_peak_test()
pip install "moving_corr_sig[notebook] @ git+https://github.com/kanchukaitis/moving_corr_sig.git"
```

## Quick start

```python
from moving_corr_sig import MovingCorrelationTest

mct = MovingCorrelationTest(seriesA, seriesB, window=15, noise="red")

std_result   = mct.std_test(iters=2000)
peak_result  = mct.peak_test(iters=2000, condition="observed")
range_result = mct.range_test(iters=2000)

print(std_result.summary())
```

### Peak-test significance thresholds for plotting

`peak_test`'s null distribution is built from the *maximum* correlation seen
anywhere in each surrogate trace, so a single threshold value (e.g. the 95th
percentile of that null distribution) already accounts for having searched
the whole record, and applies as a flat horizontal line across the entire
observed trace. This is useful when the observed moving correlation mostly
sits below the null model but has one or a few genuine excursions above it:

```python
peak_result = mct.peak_test(iters=2000, levels=(0.90, 0.95, 0.99))

peak_result.quantiles          # {'90': ..., '95': ..., '99': ...}
peak_result.threshold(0.975)   # any level, computed on demand -- not
                                # limited to what's in `levels`

# Ready-made plot: trace + threshold line(s) + highlighted exceedance points
ax, peak_result = mct.plot_peak_test(iters=2000, levels=(0.90, 0.95))
```

`plot_peak_test` requires matplotlib (`pip install moving_corr_sig[notebook]`
or just `pip install matplotlib`); the rest of the package does not.

See `moving_corr_sig_demo.ipynb` for a full walkthrough, including validation
against Gershunov et al. (2001) Table 1.

## Reference

Gershunov, A., N. Schneider, and T. Barnett, 2001. Low-frequency modulation
of the ENSO-Indian monsoon rainfall relationship: Signal or noise?,
*Journal of Climate*, 14: 2486-2492.
