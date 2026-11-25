# Evaluation data

## Manuscript tables

These files transcribe the reported cells of the manuscript tables at their printed precision. A row identifies an equation, dimension and method. RMSE is root mean squared error; RelMAE is relative mean absolute error. Lower is better.

| File | Contents |
| --- | --- |
| [Whole-function accuracy](whole_function.csv) | 160 method-level rows across 40 settings, with RMSE and RelMAE. |
| [Single-point accuracy](single_point.csv) | 60 method-level rows across 20 settings. A source cell marked NF has empty metric fields and status `numerical_failure`; it must not be treated as zero. |
| [Component ablation](component_ablation.csv) | 16 mean RMSE values with sample standard deviations. The fixed initial actor uses two inference replicates; the other rows use three training seeds. |
| [Six appendix tables](appendix_tables.tex) | Typeset table excerpts covering the MLP depth and sampling sweep, direct value-head evaluations, baseline budgets, component ablations and iteration-by-iteration results. |

## Earlier experiment summaries

[Whole-function runs](historical_whole_function.csv) and [ablation runs](historical_ablation.csv) contain run-level metrics, method identifiers, seeds and the recorded budgets or timings. The whole-function file is an earlier seven-PDE, seed-zero suite with 140 method-level rows. It uses a different scope from the manuscript table. Use the manuscript CSVs to compare the numbers displayed in the paper.

All CSV rows describe a method, setting or run. Use the evaluation example on your own inputs to generate point-level predictions locally.
