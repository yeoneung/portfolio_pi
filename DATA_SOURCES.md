# Data and conventions

## Numerical benchmarks

`market_params.npz` contains the fixed synthetic joint-market coefficients.
`metrics.json`, the N=30 / L=10 `publication_slice.npz`, and training curves
are the supplied run outputs used for the benchmark table and plots.
The exact coefficient-ODE, constrained finite-difference, neural-diagnostic,
and polynomial-certificate results are provided in `results/`.

## Empirical portfolios

The four universes are the FF10, FF12, and FF17 industry portfolios and FF25
size/book-to-market portfolios. `base_bundles/` supplies the return, macro,
Fama--French factor, and bond panels needed for the selected predictor
specifications. Return panels and the field `gross_return` in the strategy
CSV files are **excess returns**. Monthly total return is `RF + gross_return`.
Sharpe uses excess returns; geometric return, volatility, drawdown, and CRRA
CER in the manuscript use reconstructed total returns. The supplied evaluation
paths contain 239 monthly returns, from February 2000 through December 2019.

The risk-free series is supplied in the July-2025 monthly five-factor source
ZIP from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
Its SHA-256 digest is
`42492fc7fe23de2c44058e77414e1d35c451bf8d289dfff75c5b80c2804324e2`.
The extraction routine checks this digest and rejects missing evaluation dates.
The overlapping five-factor observations match the supplied factor panels in
99.6% of entries at their published precision. The RF series used when the
strategy paths were generated is not available for a byte-level comparison:
this is a provenance-matched reconstruction, not a verified identical vintage.

The macro predictors include inflation, industrial production, unemployment,
short rates, Treasury yields, and term/default spreads. The supplied panels
are fixed research inputs, not a claim of point-in-time macro-vintage data.
The submission's bootstrap/HAC calculations condition on the supplied selected
specifications; the universes overlap and are not independent replications.

Input files retain their numerical contents. Package configurations replace
machine-local path fields with portable paths; the training entry point then
generates its own inputs and resolves paths at runtime. `MANIFEST.json` records
the packaged bytes. Resolved configurations identify the supplied choice;
full retraining and a new predictor search are distinct operations.
