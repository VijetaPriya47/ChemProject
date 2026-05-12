import numpy as np
from scipy.optimize import differential_evolution


def run_genetic_algorithm(
    trained_model,
    scaler,
    bounds_dict: dict[str, tuple[float, float]],
    risk_aversion: float = 0.0,
) -> dict:
    """
    Optimize process settings via Differential Evolution.

    Objective:
    Minimize a robust negative score to maximize target output.

    Robust score (if supported by the model):
    score = mean_prediction - risk_aversion * std_prediction
    """
    if not bounds_dict:
        raise ValueError("No optimization bounds provided.")

    feature_names = list(bounds_dict.keys())
    bounds = [bounds_dict[name] for name in feature_names]

    def objective(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        try:
            mean_pred, std_pred = trained_model.predict(x_scaled, return_std=True)
            mean_pred = float(mean_pred[0])
            std_pred = float(std_pred[0])
            score = mean_pred - (float(risk_aversion) * std_pred)
            return -score
        except TypeError:
            prediction = float(trained_model.predict(x_scaled)[0])
            return -prediction

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=200,
        popsize=15,
        tol=1e-6,
        seed=42,
    )

    best_inputs = result.x
    best_inputs_scaled = scaler.transform(np.asarray(best_inputs, dtype=float).reshape(1, -1))
    try:
        mean_pred, std_pred = trained_model.predict(best_inputs_scaled, return_std=True)
        expected_mean_yield = float(mean_pred[0])
        expected_std_yield = float(std_pred[0])
    except TypeError:
        expected_mean_yield = float(trained_model.predict(best_inputs_scaled)[0])
        expected_std_yield = None

    expected_max_yield = expected_mean_yield

    return {
        "feature_order": feature_names,
        "optimal_conditions": {
            name: float(value) for name, value in zip(feature_names, best_inputs, strict=False)
        },
        "expected_max_yield": expected_max_yield,
        "expected_std_yield": expected_std_yield,
        "risk_aversion": float(risk_aversion),
        "converged": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
    }
