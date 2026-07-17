# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/) (while the
version stays below 1.0.0, breaking changes bump the minor version rather
than the major version).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-07-17

### Changed
- **Breaking:** `ar1_fit(x)` now returns `phi` alone (a plain `float`)
  instead of the tuple `(phi, sigma_e)`. `sigma_e` was computed and returned
  but never used anywhere in the package -- `red_noise_pair` derives its
  surrogates entirely from `phi_x`/`phi_y` and a target correlation, not
  from innovation variance. Update any code doing `phi, _ = ar1_fit(x)` to
  `phi = ar1_fit(x)`.
- **Breaking:** `statsmodels` is now a required dependency (previously
  optional, with a manual fallback AR(1) estimator used when it wasn't
  installed). `ar1_fit` now uses statsmodels' `AutoReg` unconditionally.
- Internal: `std_test`, `peak_test`, and `range_test` now share a common
  `_run_test` helper instead of each independently repeating the same
  ~15 lines of setup logic. No change in behavior or public API.

### Removed
- **Breaking:** `tol` and `max_tries` parameters removed from `std_test`,
  `peak_test`, and `range_test` (they remain on `white_noise_pair` and
  `red_noise_pair`, where the rejection-sampling loop they control actually
  lives). These were never overridden from their defaults anywhere in the
  package or its documentation.
- **Breaking:** `TestResult.extra` field removed. Its only use
  (`peak_test` storing `{"condition": condition}`) duplicated information
  already present in `TestResult.name`, and nothing in the package ever
  read `.extra`.
- The `[project.optional-dependencies] statsmodels` extra in
  `pyproject.toml` -- statsmodels is now a core dependency (see above).

## [0.1.0] - 2026-07-13

Initial release.

### Added
- `moving_correlation`: vectorized Pearson/Spearman moving (windowed)
  correlation.
- `std_test`: Gershunov et al. (2001) test for excess/deficient variability
  in a moving-correlation trace, conditioned on the overall correlation.
- `peak_test`: test for whether the highest windowed correlation exceeds
  chance, with `condition="zero"` (existence) or `condition="observed"`
  (elevation above background) null modes.
- `range_test`: test for whether the swing between the highest and lowest
  windowed correlation exceeds chance.
- White-noise and red-noise (AR(1)) surrogate generation
  (`white_noise_pair`, `red_noise_pair`), selectable via `noise=` on all
  three tests.
- `MovingCorrelationTest` convenience class, `TestResult.threshold()` for
  on-demand null-distribution quantiles, and `plot_peak_test()` for a
  ready-made trace-plus-threshold plot.
- Validated against Table 1 of Gershunov et al. (2001).
