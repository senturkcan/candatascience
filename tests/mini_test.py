def fit_encode_and_scale(
    x_train, x_test,
    categorical_cols=None,
    numeric_cols=None,
    unknown_token="__unseen__",
    missing_token="__missing__",
    scaler=None
):
    """
    Fit OneHotEncoder + Scaler on x_train, apply to x_test.
    Handles missing/unseen categories safely.
    Returns encoded DataFrames and fitted ColumnTransformer.
    """
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from sklearn.impute import SimpleImputer

    scaler = scaler or RobustScaler()
    X_tr, X_te = x_train.copy(), x_test.copy()

    # Infer column types if not provided
    categorical_cols = categorical_cols or X_tr.select_dtypes(["object", "category"]).columns.tolist()
    numeric_cols = numeric_cols or X_tr.select_dtypes(include=[np.number]).columns.tolist()

    # Prepare categories for OHE (train only)
    categories_for_ohe, known_sets = [], {}
    for col in categorical_cols:
        X_tr[col] = X_tr[col].fillna(missing_token).astype(str)
        unique_vals = sorted(set(X_tr[col]) | {missing_token, unknown_token})
        categories_for_ohe.append(unique_vals)
        known_sets[col] = set(unique_vals) - {unknown_token}

    # Handle test set
    for col in categorical_cols:
        X_te[col] = X_te[col].fillna(missing_token).astype(str)
        X_te[col] = X_te[col].apply(lambda v: v if v in known_sets[col] else unknown_token)

    # Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ]) if numeric_cols else None

    categorical_pipeline = Pipeline([
        ("ohe", OneHotEncoder(categories=categories_for_ohe, handle_unknown="ignore", sparse=False))
    ]) if categorical_cols else None

    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ]
    transformers = [t for t in transformers if t[1] is not None]

    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.fit(X_tr)

    # Transform and wrap into DataFrames
    X_train_enc = pd.DataFrame(preprocessor.transform(X_tr),
                               columns=preprocessor.get_feature_names_out(),
                               index=x_train.index)
    X_test_enc = pd.DataFrame(preprocessor.transform(X_te),
                              columns=preprocessor.get_feature_names_out(),
                              index=x_test.index)

    return X_train_enc, X_test_enc, preprocessor
