# Final presentation revision

The manuscript now frames two findings: attribution representation controls detection sensitivity, while unsupervised drift scores give uneven conditional-risk rankings. Current confirmation results provide the empirical motivation throughout.

The final paper reports the covariate/boundary interaction comparison, W50/W250 results, weight-only/quantile-only/directional ablations, ADWIN detection counts and observed threshold transfer. Row-best unrounded means and exact ties are bold. AUROC is displayed to four decimals to distinguish rounded near-ties. References 1 and 2 use explicit, spaced LaTeX em dashes. Figure 1 pairs Hyperplane/MLP with Adult/XGB, with increased horizontal spacing and embedded TrueType fonts.

The requested endpoint-saturation explanation was checked against the mapping-calibration segment. Five of 6000 shifted XGB predictions across the five fits (0.083%) exceed the mapping maximum. The auxiliary map has 327–347 distinct values, while the full CP score retains 1198–1200 distinct values. The logged weight-only ablation gives AURC 0.7671, quantile-only 0.7321, and directional-naive 0.7194, compared with full CP 0.7701. The manuscript uses those measured contrasts to explain the observed gap. These mapping diagnostics are explicitly post hoc; all primary confirmation outcomes remain fixed.

This pass uses existing results only. Model-fitting calls: 0. The four frozen result archives and scientific engine retain their original SHA256 checksums.
