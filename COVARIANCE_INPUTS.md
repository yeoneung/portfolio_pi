# Covariance inputs for the empirical solvers

The manuscript uses conditional return risk and return--state innovation
covariance as inputs to each control problem. Covariance-filter states are
not additional PDE states. The selected configurations determine the estimator.

## Selected specifications

All four selected configurations use automatic cross-covariance selection.

| Universe | Configured covariance model | Joint return--state estimator |
|---|---|---|
| FF10 | asset_dcc | DCC |
| FF12 | asset_adcc | ADCC |
| FF17 | asset_adcc | ADCC |
| FF25 | constant | Sample residual second moments |

The configurations are under
code/empirical_src_bundle/comparative/*/selected_by_pipinn/rank_001/.
Model dispatch is in
[experiments.py](code/empirical_src_bundle/dynalloc_v2/experiments.py).
The constant model is constant within a refit, not across the full sample.

## Joint covariance supplied to the PDE

Let \(\varepsilon_j^r\) be the return residual and \(\varepsilon_j^Z\) the
state-VAR innovation in a refit window. Without a dynamic filter, the joint
estimator uses

\[
 \frac{1}{\max(n-1,1)}\sum_{j=1}^n
 \begin{pmatrix}\varepsilon_j^r\\\varepsilon_j^Z\end{pmatrix}
 \begin{pmatrix}\varepsilon_j^r\\\varepsilon_j^Z\end{pmatrix}^{\!\top}.
\]

These are residual second moments as implemented; this branch applies no
additional sample-mean subtraction. With a dynamic filter, the stacked
residuals are centered using the refit-window mean, and the filter supplies
the current joint matrix. Its asset, cross, and state blocks supply the PDE
coefficients consistently. See
[transition.py](code/empirical_src_bundle/dynalloc_v2/transition.py).

The field covariance_train_mode = dcc_current requests the current asset
block from this estimator. It does not force DCC when the estimator is
constant. For FF25 it returns the sample asset residual block, not the
separate factor-model covariance prediction. Coefficients are frozen during
each PDE solve.

## DCC and ADCC recursions

For a centered innovation \(e_t\), standardize using the pre-update variance:
\(z_t=D_{t-1}^{-1}e_t\). Before numerical regularization, the updates are

\[
 h_{i,t}=\lambda h_{i,t-1}+(1-\lambda)e_{i,t}^2,\qquad
 Q_t=(1-a-b)\bar Q+a z_tz_t^\top+bQ_{t-1}.
\]

The variance filter is EWMA, not a fitted positive-intercept GARCH recursion.
For ADCC, with \(n_t=\min(z_t,0)\) componentwise and \(\bar N\) the
refit-window mean negative-innovation outer product, the update adds
\(g(n_tn_t^\top-\bar N)\).

Normalize by
\(R_t=\operatorname{diag}(Q_t)^{-1/2}Q_t
\operatorname{diag}(Q_t)^{-1/2}\), then form \(D_tR_tD_t\).
The supplied settings are \(\lambda=0.97\), \(a=0.02\), \(b=0.97\),
and \(g=0.005\) for ADCC. Correlation shrinkage is 0.1 and the
variance/eigenvalue floor is \(10^{-6}\). The routines symmetrize matrices
and floor eigenvalues during construction; the PDE backend additionally
symmetrizes its asset block with a \(10^{-10}\) floor. The separate asset
covariance model has diagonal shrinkage 0.1; this is not an extra step in
the joint-block selector.

Initialization and the exact operation order are in
[covariance.py](code/empirical_src_bundle/dynalloc_v2/covariance.py) and
[transition.py](code/empirical_src_bundle/dynalloc_v2/transition.py);
the selector is in
[pipinn_backend.py](code/empirical_src_bundle/dynalloc_v2/pipinn_backend.py).
This documents the supplied estimator, not a statistical consistency claim
or a diffusion-limit theorem.
