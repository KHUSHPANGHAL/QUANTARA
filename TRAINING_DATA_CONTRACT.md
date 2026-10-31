# Training data contract

1. Choose exactly one horizon folder (`horizon_1d`, `horizon_5d`, or `horizon_20d`).
2. Read X and y from the same split; preserve row order. Do not sort or shuffle X, y, and metadata independently.
3. Use all columns in X as input features; X contains only the 20 approved numeric features.
4. For classification, fit against `Target_{h}d`. For regression, fit against `forward_return_{h}d`. Do not use both target columns as features.
5. Fit preprocessing (imputation, scaling, feature selection) only on training data; transform validation/test using the fitted training pipeline.
6. Use validation data for model selection. Evaluate the selected model on test once at the end.
7. Do not join the audit-only master dataset into X without explicitly selecting the feature list.
8. The 5-session and 20-session class thresholds are initial modelling assumptions; tune only using training/validation.
9. This dataset is not investment advice. Test metrics do not guarantee live-market performance.
