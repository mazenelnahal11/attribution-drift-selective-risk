# Path A+ protocol 2.0

## Estimands and score definitions

The classifier stays fixed, predicting class 1 when p >= 0.5. False-negative cost is 5; false-positive cost is 1. Reported realized risk and AURC use these original cost units. Expected-cost ranking uses c=w*u, with u=min(p,1-p) and w=1 for predicted class 0 or 0.2 for predicted class 1.

The scalar D is half base-2 Jensen–Shannon divergence of normalized absolute attribution mass plus half weighted top-three pairwise rank discordance. Zero attribution vectors map to uniform mass. The signed discrepancy V=A-B retains signs, magnitudes and d coordinates.

Three source calibration segments, each 512 observations, have separate roles:

1. Mapping: fit the right-continuous empirical CDF of D, then the inverse empirical CDF of u. Apply the inverse by ceil(n*q)-1, clamped at both endpoints.
2. Detection: calibrate one threshold per representation and window from 199 moving-block bootstrap resamples, block length 20, using the higher 95th percentile. Resample indices are shared across representations.
3. Selection: fit higher score quantiles for 10% and 20% review fractions. Score exceedance triggers deferral; ties are accepted.

The naive score is (1-g)*c+g*D. Here g=max(0,1-q/delta) when the scalar W1 statistic exceeds q, otherwise zero. The CP score is w*((1-g_directional)*u+g_directional*matched_D), with g_directional=g only when the current window mean D exceeds its reference mean. Both statistics use features and fixed model/explanation outputs. Stream labels enter the separate evaluation layer.

Window sizes are 50, 100 and 250; stride is floor(W/4), with W=100 primary. Actions update when a complete window ends, incorporating the current observation. They hold between updates. The first W-1 observations use zero gamma.

The CP score stays in [0,w/2], and equals c if its trigger is zero or mapped D equals u. Preserving the weight and range permits rank changes. An explicit counterexample is p=(0.11,0.60), c=(0.11,0.08), mapped_D=(0.01,0.50), gamma=1, CP=(0.01,0.10). Exact conditional cost orders Bayes-optimal acceptance for a fixed classifier; an auxiliary score needs additional risk information to improve that ordering.

## Models and attribution computations

Each unit uses 4096 training, 384 gate, 256 reference and 1536 calibration observations. The source standardizer and a common 64-row background are fitted using training observations. Source checks assess balanced accuracy >=0.60, 95th-percentile normalized completeness residual <=0.05, median local weighted R2 >=0.10, D standard deviation >=0.0001, and finite outputs. Development stationary scalar alarm fractions at W100 must be <=0.20. Effect direction is retained for every score.

XGBoost: 120 trees, depth 3, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, reg_lambda 1, histogram tree method. MLP: 64/64/32 hidden widths, SiLU, AdamW learning rate 0.001 and weight decay 0.0001, 30 epochs, batches 256. The original classifier threshold 0.5 is held fixed for every gate.

A uses interventional TreeSHAP in logit space for XGB and 64-point Gauss–Legendre IG from the background mean for MLP. B is a local ridge surrogate: 500 antithetic Gaussian perturbations (sigma 0.5), weights exp(-||delta||²/(d*0.75²)), ridge 0.001*N with the intercept unpenalized, slopes multiplied by displacement from the background mean. A second local fit estimates within-procedure sampling disagreement.

Scalar monitors include D, its JSD and rank components, within-surrogate disagreement, entropy, MSP uncertainty, symmetric-logit energy, expected cost and source distance. The vector monitors A, B and A-B use identical mean featurewise W1 statistics. Input monitors use maximum marginal KS and 64-feature random-Fourier RBF MMD with reference-median bandwidth. Confidence ADWIN uses its native delta 0.002.

## Data, new confirmation observations and fit seeds

The fitting seeds are 11, 29, 47, 71 and 101. Data seed 20261110, explanation seed 801 and monitor seed 1701 are fixed across fits. The 10-unit development grid and 50-unit confirmation grid use different deployment observations. All source roles are disjoint from deployment identifiers. Cross-scenario reuse is explicit pairing.

- SEA: three uniform inputs, threshold rule with 5% label noise. The observable arm transforms the first feature by a square root at the midpoint.
- Hyperplane: ten features; the first two begin positively correlated and reverse correlation gradually from midpoint to three-quarter position. The observable arm preserves the conditional label rule with 5% label noise.
- Adult: six numeric inputs; full-row deduplication precedes splitting. Hash-order even/odd official training rows supply development/confirmation sources. Revised confirmation deployment uses previously unused official test identifiers. IDs examined in any original development or confirmation role are excluded from revised deployment. The manifest fixes those exclusions before the revised outcomes. Within each controlled stream, weighted post-onset sampling is without replacement and depends on numeric features.
- Electricity: original chronology, seven inputs after excluding date. Revised development starts at 8000 and confirmation at 30000. Confirmation deployment is zero-based rows [36272,38672). Source counts and positions are recorded per role.
- Interaction: six normal features, source correlation rho=0.95 between x0 and x1. Source logit is 2*(x0+x1+1.5*(x0*x1-rho)+0.25*x2). Correlation reverses to -rho at the midpoint. The boundary arm adds 1.5*exp(-((x0+x1)/0.5)²)*tanh(2*(x0-x1)) inside the doubled logit. Covariate and boundary arms share inputs and label uniforms. True posteriors are recorded for evaluation. Stationary and fixed-input changed-label arms complete the controls. The generator is independent of fitted models and explanations.

Controlled confirmation streams have 2400 observations each. The primary action endpoint uses fully shifted observations; Electricity uses the entire natural stream. Risk curves admit tied-score groups together. AURC sums risk times successive coverage increments. Fixed-threshold costs are reported together with realized coverage. Drift AUROC compares wholly pre-onset and wholly post-transition windows. Overlapping windows and shared cases are retained in the interpretation.

Component ablations keep only cost preservation, only quantile matching, a two-sided cost+quantile score, or a directional unscaled score. Supplementary tables retain all arms and all window sizes.

## Preservation and verification

Original scientific engine SHA256: `6b820624abc4ecca124d9b2ed884f896caf313c731035d0fe34d13f86628a2bd`.

Revised scientific engine SHA256: `53a0d22652fdff2683dc9053f8ad072f09958b79b518f0ba5edb63ccebcbf4e4`.

The original and revised archives remain immutable. File manifests, configurations, source splits, raw A/B/B2 arrays, fitted checkpoints, scalar/vector statistics, bootstrap samples, calibration cutoffs, per-row scores, costs and timing support independent reconstruction. A separate audit reproduces every XGBoost fit with the intended sampling settings and replays every saved model's predictions. All ten benchmark/model groups have five distinct fitted models and probability arrays on one common case set.

The frozen scientific code is preserved byte for byte. The portable CPU launcher accepts the same public PyTorch release 2.8.0 with an optional CPU wheel build suffix, records the full installed version, and adapts confirmation process count to CPU cores. Manifest line endings are normalized explicitly during cross-platform identity checks. Scientific settings retain their frozen identities.

The manuscript reports the verified outcomes, including ties and higher-risk outcomes. False Assurance and Contextual Reversals remain separate studies.
