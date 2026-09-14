# Source code

## Data preprocessing

```bash
python src/s1_preprocessing.py
```

- **`s1_preprocessing.py`** reads the `D.1_drained.csv`, `D.2_drained.csv`, `U_undrained.csv`, `real_cases.csv` files, 
  merges the `D.1_drained.csv` and `D.2_drained.csv` datasets into `D_drained.csv`. It creates 80% training and 20% test
  partitions and prepares `dataset/real_cases.csv` for inference.

```text
dataset/
|-- D.1_drained.csv
|-- D.2_drained.csv
|-- D_drained.csv
|-- U_undrained.csv
|-- real_cases.csv
|-- train/
|   |-- <condition>_fos.csv
|   |-- <condition>_zwu.csv
|   `-- <condition>_zwd.csv
`-- test/
    |-- <condition>_fos.csv
    |-- <condition>_zwu.csv
    `-- <condition>_zwd.csv
```

`<condition>` is `drained` or `undrained`.

- The `fos` datasets are used to predict the **factor of safety** (`FoS`) and
  **depth of the slip surface** ($z_s$).
- The `zwu` datasets are used to predict the **final upstream piezometric
  depth** ($z_{wu}^{\mathrm{final}}$). Samples with
  $z_{wu}^{\mathrm{init}} = 0$ are excluded.
- The `zwd` datasets are used to predict the **final downstream piezometric
  depth** ($z_{wd}^{\mathrm{final}}$). Samples with
  $z_{wd}^{\mathrm{init}} = 0$ are excluded.

## Feature selection

```bash
python src/s2_feature_selection.py
```

- **`s2_feature_selection.py`** performs sequential forward feature selection
  for every target, slope condition, and regressor.
  It repeats selection considering stratified cross-validation partitions and then evaluates the
  models using R² and MAE.

Outputs:

- `results/selected_features.json` with the selected feature sets;
- `feature_freq.csv` files with feature-selection frequencies;

## Hyperparameter optimization

```bash
python src/s3_hyperparam_opt.py
```

- **`s3_hyperparam_opt.py`** starts from the feature sets produced by `s2_feature_selection.py`.
  It combines the grid search with repeated nested cross-validation,
  evaluates the selected hyperparameter combination on the outer folds, and selects a
  final hyperparameter combination.

Outputs:

- `scores_r2.csv` and `scores_mae.csv` with cross-validation scores;
- `<ModelName>_test.pkl` and `<ModelName>_inference.pkl` model bundles.

## Model evaluation

```bash
python src/s4_boxplot.py
python src/s5_regplot.py
```

- **`s4_boxplot.py`** reads the cross-validation score and ranks models
  by their median R² or MAE. It generates boxplots for the prediction targets
  and pairwise Wilcoxon signed-rank comparison matrices. The statistical
  results use Holm correction for multiple comparisons and are saved in
  `fig/holdout/`.

- **`s5_regplot.py`** loads the true and predicted hold-out values.
  It computes R² and MAE and generates regression plots
  comparing observations with model predictions. The resulting figures are
  organized by slope condition and target in `fig/holdout/`.

## Explain model predictions

```bash
python src/s6_shap_test.py
python src/s7_shap_inference.py
```

- **`s6_shap_test.py`** computes SHAP explanations for the hold-out datasets.
  It produces global beeswarm plots for the test set.
  It also selects representative slope cases and
  generates local waterfall plots showing how each feature contributes to a
  prediction. The resulting figures are saved in `fig/shap_test/`.

- **`s7_shap_inference.py`** explains predictions for the external cases stored
  in `dataset/real_cases.csv`. It uses the combined training and test data as
  the SHAP background distribution. A waterfall plot is generated for every case and target.
  The resulting figures are saved in `fig/shap_inference/`

## Run the fuzzy assessment

```bash
python src/s8_fis.py
```

- **`s8_fis.py`** predicts the four target quantities for each external case
  using the saved inference models. The predicted slip-surface depth and mean
  final piezometric depth are passed to Mamdani fuzzy systems that estimate the
  effectiveness of the available mitigation guidelines. The script saves
  predictions, final rankings, and crisp and fuzzy surface figures in `fig/fis/`


## Shared configuration and utilities

- **`config.py`** contains paths, target names, rainfall variables,
  normalization settings, model definitions, hyperparameter grids, labels,
  and plot colors.

- **`assets.py`** contains helpers shared by the training and SHAP
  scripts. It creates the configured feature scaler, selects an appropriate
  SHAP explainer, and converts the results into SHAP
  explanation objects.


<!-- ## SHAP plotting utilities

- **`plots/beeswarm.py`** implements the global SHAP beeswarm visualization.
  It orders features by importance, distributes samples to show their density,
  and colors points according to feature values. The function also supports
  display limits, clustering, custom ordering, and color-bar formatting.

- **`plots/waterfall.py`** implements local SHAP waterfall plots for individual
  predictions. It shows how positive and negative feature contributions move
  the model output from the expected value to the final prediction. Less
  important features can be grouped when the display limit is exceeded.

- **`plots/assets/colorconv.py`** provides the color-space conversions used to
  construct the project palettes. The routines convert between LCH, LAB, XYZ,
  and RGB representations. They are included locally so that the plotting code
  does not require the full scikit-image package.

- **`plots/assets/colors.py`** defines the continuous and fixed color palettes
  used by the SHAP figures. It builds perceptually varied red-blue scales and
  related color constants through the conversion functions. These palettes
  keep global and local explanation figures visually consistent.

- **`plots/assets/utils.py`** contains common formatting, ordering, and
  clustering helpers used by the custom plots. It handles SHAP operation
  histories, color conversion, feature sorting, hierarchical tree traversal,
  node merging, and dendrogram coordinates. These functions support both plot
  layout and readable value labels.

## Outputs

- `dataset/train/` and `dataset/test/`: prepared training and test datasets;
- `results/`: selected features, evaluation scores, serialized models,
  external-case predictions, and feature summaries;
- `fig/holdout/`: score comparisons and observed-versus-predicted plots;
- `fig/shap_test/` and `fig/shap_inference/`: SHAP explanation figures;
- `fig/fis/`: fuzzy rankings and response surfaces.

Files ending in `_test.pkl` support hold-out evaluation. Files ending in
`_inference.pkl` support prediction on new cases.
 -->