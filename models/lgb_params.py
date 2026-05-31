"""
自适应 LightGBM 参数 — 防止样本少时 min_data_in_leaf 过大导致无法分裂
"""
def get_lgb_params(n_rows, n_codes, n_features, mode="cross_section"):
    if mode == "cross_section":
        min_data_in_leaf = max(3, n_codes // 20)
        num_leaves = min(15, max(7, n_codes // 8))
        max_depth = 4
    else:
        if n_rows < 1000: min_data_in_leaf = max(3, n_rows // 50); num_leaves = 15; max_depth = 4
        elif n_rows < 20000: min_data_in_leaf = 30; num_leaves = 31; max_depth = 5
        else: min_data_in_leaf = 100; num_leaves = 31; max_depth = 6
    return {
        "objective": "regression", "n_estimators": 500, "learning_rate": 0.03,
        "num_leaves": num_leaves, "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf, "feature_fraction": 0.7,
        "bagging_fraction": 0.8, "bagging_freq": 3,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "random_state": 42, "verbosity": -1,
    }
