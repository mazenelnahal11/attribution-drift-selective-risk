# From Attribution Drift to Selective Action

Path A+ research package for the ESANN 2027 study. The revised development grid (10 fits) and confirmation grid (50 fits) have completed. The confirmation records passed 124,263 independent checks of saved-model predictions, row scores, risk metrics and fit diversity, plus 460,600 source-calibration checks: 584,863 checks in total.

The five fit seeds produce distinct checkpoints and probability arrays in every benchmark/model group. Data, explanation perturbations and monitor resamples are shared across those fits. Sample seed SD describes fitted-model variation on common observations.

## Read the results

- `analysis/RESULTS.md`: both updated tables and paired outcome counts.
- The companion local research package supplies the six-page manuscript, complete LaTeX source and official class.
- `analysis/figures/figure1.pdf`: vector figure with embedded TrueType fonts. The PNG export is 600 DPI; both panels have source CSVs.
- `EXACT_FUNCTIONS.md`: the exact implementations of scalar disagreement, featurewise Wasserstein monitoring, expected cost, quantile matching and action scores.
- `PROTOCOL.md`: scientific settings, split rules and the boundary interpretation.

On the original three controlled benchmark designs with fresh observations, naive blending raises cost AURC in 22/30 fits. Cost-preserved blending raises it in 17, lowers it in 2 and ties in 11. In the interaction boundary arm, CP raises AURC for all five XGB fits and equals expected cost for all five MLP fits. The signed discrepancy vector improves detector AUROC over scalar disagreement in five of eight controlled benchmark/model groups.

The paper develops these outcomes as evidence about detection, risk ordering and transferred review thresholds. The mathematical construction preserves explicit class weights and source scale; its rank changes are evaluated empirically.

## Colab: open and Run all

Use a **CPU runtime**. Setup, downloads, checksums, computation, auditing and the single results ZIP export are preconfigured.

1. [Development](https://colab.research.google.com/github/mazenelnahal11/attribution-drift-selective-risk/blob/main/01_Path_A_Plus_Development.ipynb): reproduce 10 development fits and their freeze.
2. [Confirmation](https://colab.research.google.com/github/mazenelnahal11/attribution-drift-selective-risk/blob/main/02_Path_A_Plus_Confirmation.ipynb): reproduce all 50 fits, automatically loading the completed development freeze.
3. [Verified analysis](https://colab.research.google.com/github/mazenelnahal11/attribution-drift-selective-risk/blob/main/03_Path_A_Plus_Verified_Analysis.ipynb): audit the completed confirmation archive and regenerate the tables and Figure 1.

The completed results are already supplied. Notebook 03 reproduces their analysis; notebook 02 performs a fresh full rerun. The CPU launcher records its full PyTorch build string and supports the `+cpu` wheel suffix of public release 2.8.0. It adapts the confirmation process count to available cores, retaining the scientific configuration.

## Source and reproducibility

`source/engine.py` is the byte-preserved scientific engine. Its SHA256 is `53a0d22652fdff2683dc9053f8ad072f09958b79b518f0ba5edb63ccebcbf4e4`. The source mapping, detector calibration and selection calibration each use 512 different observations.

The standalone independent audit is `source/audit_v2.py`. `source/audit_calibration.py` additionally reconstructs every bootstrap resample and every source selection-window statistic. `source/analyze_v2.py` generates both tables, all supplementary summaries and Figure 1. `source/portable_runner.py` preserves the engine while providing cross-platform CPU-build validation.

With the pinned requirements and CPU PyTorch 2.8.0 installed:

```text
python source/portable_runner.py confirmation PATH_TO_DEVELOPMENT_ZIP
python source/audit_v2.py EXTRACTED_CONFIRMATION_DIRECTORY --full
python source/audit_calibration.py EXTRACTED_CONFIRMATION_DIRECTORY
python source/analyze_v2.py EXTRACTED_CONFIRMATION_DIRECTORY analysis
```

The [protocol-v2.0 release](https://github.com/mazenelnahal11/attribution-drift-selective-risk/releases/tag/protocol-v2.0) supplies the raw revised archives and preserved historical archives. `SHA256SUMS.txt` identifies each exact asset. Original version 1 results remain a separate historical baseline. `historical/` contains its source and summary; the revised naive comparator is recomputed on each new checkpoint and case.

This tabular study is separate from False Assurance and the submitted Contextual Reversals study. The protocol and bibliography attribute the shared explanation methods to their primary sources.
