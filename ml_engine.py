import sys

def _force_user_site_mpl_toolkits() -> None:
    """
    Force `mpl_toolkits` to resolve from user site-packages.

    On some systems, `mpl_toolkits` is auto-imported from system dist-packages
    at interpreter startup, which can later conflict with a pip-installed
    `matplotlib` in user site-packages. The resulting version-mix can crash
    Matplotlib during import (e.g., missing private Axis APIs).
    """
    import importlib
    import importlib.machinery
    import site

    # Remove any preloaded system mpl_toolkits modules.
    for name in list(sys.modules.keys()):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]

    user_site = site.getusersitepackages()
    spec = importlib.machinery.PathFinder.find_spec("mpl_toolkits", [user_site])
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["mpl_toolkits"] = module
        spec.loader.exec_module(module)
        importlib.invalidate_caches()


_force_user_site_mpl_toolkits()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler


def willmott_index_of_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Willmott's Index of Agreement (WI).

    Formula:
    WI = 1 - [sum((P - O)^2) / sum((|P - O_bar| + |O - O_bar|)^2)]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_mean = np.mean(y_true)
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((np.abs(y_pred - y_true_mean) + np.abs(y_true - y_true_mean)) ** 2)

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return float(1.0 - (numerator / denominator))


def train_evaluate_gpr(df: pd.DataFrame, input_cols: list[str], target_col: str) -> dict:
    """
    Train and evaluate GPR.

    Strategy:
    - For small experimental datasets (default: N <= 80): LOOCV with out-of-fold predictions.
    - For larger datasets: switch to an 80/20 train-test split (LOOCV becomes impractical).
    - For very large datasets: subsample for tractable GPR training (GPR scales poorly with N).
    """
    if not input_cols:
        raise ValueError("At least one input feature is required.")

    model_df = df[input_cols + [target_col]].copy()
    model_df[input_cols] = model_df[input_cols].apply(pd.to_numeric, errors="coerce")
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna()

    if len(model_df) < 5:
        raise ValueError("Not enough valid numeric rows after cleaning. Need at least 5 rows.")

    x = model_df[input_cols].values
    y = model_df[target_col].values

    n_samples = int(len(y))
    strategy = "loocv" if n_samples <= 80 else "train_test"

    # Subsample for computational tractability on large datasets.
    max_gpr_samples = 400
    used_indices = np.arange(n_samples)
    if n_samples > max_gpr_samples:
        rng = np.random.default_rng(42)
        used_indices = rng.choice(n_samples, size=max_gpr_samples, replace=False)
        used_indices = np.sort(used_indices)
        x = x[used_indices]
        y = y[used_indices]
        n_samples = int(len(y))
        strategy = "train_test_subsample"

    if strategy == "loocv":
        loo = LeaveOneOut()
        y_pred = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in loo.split(x):
            x_train_fold = x[train_idx]
            x_test_fold = x[test_idx]
            y_train_fold = y[train_idx]

            fold_scaler = StandardScaler()
            x_train_fold_scaled = fold_scaler.fit_transform(x_train_fold)
            x_test_fold_scaled = fold_scaler.transform(x_test_fold)

            fold_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            fold_model = GaussianProcessRegressor(
                kernel=fold_kernel,
                normalize_y=True,
                random_state=42,
                n_restarts_optimizer=2,
            )
            fold_model.fit(x_train_fold_scaled, y_train_fold)
            y_pred[test_idx[0]] = float(fold_model.predict(x_test_fold_scaled)[0])

        # Fit final model on all used data.
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_train_scaled = x_scaled
        x_test_scaled = x_scaled
        y_test = y

        final_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        model = GaussianProcessRegressor(
            kernel=final_kernel,
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=2,
        )
        model.fit(x_scaled, y)
        model.x_test_for_shap_ = x_scaled
        model.y_for_explainability_ = y
        n_folds = int(len(y))
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        final_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        model = GaussianProcessRegressor(
            kernel=final_kernel,
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=2,
        )
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        model.x_test_for_shap_ = x_test_scaled
        model.y_for_explainability_ = y_test
        n_folds = None

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, y_pred) * 100.0),
        "MdAE": float(median_absolute_error(y_test, y_pred)),
        "WI": float(willmott_index_of_agreement(y_test, y_pred)),
    }

    abs_errors = np.abs(np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float))

    return_data = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "abs_errors": abs_errors,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "cleaned_rows": int(len(model_df)),
        "cv_strategy": strategy,
        "used_rows_for_training": int(n_samples),
    }
    if n_folds is not None:
        return_data["n_folds"] = n_folds
    return return_data


def _fallback_permutation_importance_plot(
    model,
    x_scaled: np.ndarray,
    feature_names: list[str],
):
    """Build a fast fallback feature-importance plot using permutation importance."""
    y_ref = np.asarray(getattr(model, "y_for_explainability_", np.zeros(len(x_scaled))), dtype=float)
    importances = permutation_importance(
        model,
        x_scaled,
        y_ref,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    order = np.argsort(importances.importances_mean)[::-1]
    ordered_names = np.array(feature_names)[order]
    ordered_values = importances.importances_mean[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered_names[::-1], ordered_values[::-1], color="#2C7FB8")
    ax.set_title("Permutation Feature Importance (Fallback)")
    ax.set_xlabel("Importance (mean decrease in score)")
    ax.set_ylabel("Features")
    fig.tight_layout()
    top_feature = str(ordered_names[0]) if len(ordered_names) > 0 else None
    return fig, top_feature


def generate_shap_plot(model, X_scaled, feature_names: list[str]):
    """
    Generate explainability plot for GPR using SHAP KernelExplainer with safeguards.

    Notes:
    - Uses SHAP beeswarm when feasible.
    - Falls back to permutation importance if SHAP is likely too slow or fails.
    """
    background = np.asarray(X_scaled, dtype=float)
    explanation_data = np.asarray(getattr(model, "x_test_for_shap_", background), dtype=float)

    n_samples, n_features = explanation_data.shape
    estimated_kernel_work = n_samples * n_features * max(1, len(background))
    max_kernel_work = 12000

    if estimated_kernel_work > max_kernel_work:
        fig, top_feature = _fallback_permutation_importance_plot(
            model,
            explanation_data,
            feature_names,
        )
        return fig, "permutation", top_feature

    try:
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(
            explanation_data,
            nsamples=min(120, max(30, n_features * 12)),
        )

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[0]
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
        top_feature_idx = int(np.argmax(mean_abs_shap))
        top_feature = feature_names[top_feature_idx]

        plt.figure(figsize=(9, 5))
        shap.summary_plot(
            shap_values,
            explanation_data,
            feature_names=feature_names,
            show=False,
        )
        return plt.gcf(), "shap", top_feature
    except Exception:
        fig, top_feature = _fallback_permutation_importance_plot(
            model,
            explanation_data,
            feature_names,
        )
        return fig, "permutation", top_feature
