# Path A+ verified confirmation outcomes

Five fitted-model seeds (11, 29, 47, 71, 101) on common data, explanation perturbations and monitor resamples. Cost is 5 for a false negative and 1 for a false positive. Lower cost AURC is better. Entries show mean (sample seed SD).

|Benchmark|Model|Expected cost|Naive unscaled|Cost-preserved|
|---|---|---|---|---|
|SEA|XGB|0.0427 (0.0009)|0.0440 (0.0013)|0.0433 (0.0009)|
|SEA|MLP|0.0523 (0.0014)|0.0522 (0.0017)|0.0521 (0.0017)|
|Hyperplane|XGB|0.1433 (0.0185)|0.1471 (0.0198)|0.1438 (0.0194)|
|Hyperplane|MLP|0.1265 (0.0068)|0.1532 (0.0159)|0.1265 (0.0068)|
|Adult|XGB|0.3289 (0.0014)|0.3595 (0.0113)|0.3452 (0.0067)|
|Adult|MLP|0.3464 (0.0042)|0.3459 (0.0044)|0.3504 (0.0048)|
|Interaction|XGB|0.7580 (0.0097)|0.7194 (0.0117)|0.7701 (0.0106)|
|Interaction|MLP|0.2964 (0.0268)|0.3288 (0.0295)|0.2964 (0.0268)|
|Electricity|XGB|0.4386 (0.0044)|0.4695 (0.0028)|0.4378 (0.0041)|
|Electricity|MLP|0.3416 (0.0100)|0.3455 (0.0052)|0.3434 (0.0105)|

The interaction row uses the local conditional-boundary arm. Other controlled rows use fully shifted observations. Electricity uses the complete chronological deployment stream. The matched interaction covariate-only control is supplied separately.

## Table 2: window AUROC at W=100

|Benchmark|Model|Scalar D|Signed A-B|A|B|Entropy|KS|MMD|
|---|---|---|---|---|---|---|---|---|
|SEA|XGB|0.522|1.000|1.000|0.986|0.882|0.999|1.000|
|SEA|MLP|0.564|0.980|0.978|0.957|0.517|0.999|1.000|
|Hyperplane|XGB|0.622|0.605|0.418|0.638|1.000|0.419|0.862|
|Hyperplane|MLP|0.992|1.000|1.000|1.000|0.986|0.419|0.862|
|Adult|XGB|0.943|0.785|0.758|0.729|0.955|0.568|0.639|
|Adult|MLP|0.727|0.921|0.783|0.825|0.963|0.568|0.639|
|Interaction|XGB|0.981|0.570|0.773|0.784|1.000|0.750|1.000|
|Interaction|MLP|0.992|0.999|1.000|1.000|1.000|0.750|1.000|

Higher window AUROC is better. Transition-overlapping windows are excluded. The interaction covariate-only and boundary arms have identical inputs and monitoring statistics. Source-calibrated thresholds and stationary alarm rates are in the accompanying tables.

Independent audit: 124,263 passed checks, including every saved checkpoint and independently refitted XGBoost models.

## Paired outcome counts

- Original three controlled benchmark designs on fresh observations; naive_unscaled: 4 lower, 22 higher, 4 tied AURCs.
- Original three controlled benchmark designs on fresh observations; cost_preserved: 2 lower, 17 higher, 11 tied AURCs.
- Interaction boundary arm; naive_unscaled: 5 lower, 5 higher, 0 tied AURCs.
- Interaction boundary arm; cost_preserved: 0 lower, 5 higher, 5 tied AURCs.
- All five benchmarks; naive_unscaled: 10 lower, 36 higher, 4 tied AURCs.
- All five benchmarks; cost_preserved: 5 lower, 25 higher, 20 tied AURCs.

Source calibration reconstruction: 460,600 additional checks passed across every bootstrap replicate and source selection-window statistic.
