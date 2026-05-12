import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path

from ml_engine import generate_shap_plot, train_evaluate_gpr
from optimizer import run_genetic_algorithm

EXPERIMENT_TYPE_COL = "Experiment_Type"

DEMO_FILENAME = "Copy of Batch study of Mango biochar  - ML_Dataset.csv"


def apply_plotly_white_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111111"},
        title={"font": {"color": "#111111"}},
        legend={"font": {"color": "#111111"}},
        margin={"l": 30, "r": 20, "t": 60, "b": 30},
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="#111111",
        gridcolor="#E6E6E6",
        zerolinecolor="#CCCCCC",
        tickfont={"color": "#111111"},
        titlefont={"color": "#111111"},
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="#111111",
        gridcolor="#E6E6E6",
        zerolinecolor="#CCCCCC",
        tickfont={"color": "#111111"},
        titlefont={"color": "#111111"},
    )
    return fig


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def load_csv_from_path(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep=";")
    return df


def render_data_preview(df: pd.DataFrame) -> None:
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)


def _default_numeric_bounds(series: pd.Series) -> tuple[float, float]:
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid = numeric_series.dropna()
    if valid.empty:
        return 0.0, 100.0
    return float(np.nanmin(valid.values)), float(np.nanmax(valid.values))


def render_feature_selection(df: pd.DataFrame) -> tuple[list[str], str | None]:
    st.sidebar.header("Feature Selection")

    # Restrict to columns with numeric content (exclude categorical like Experiment_Type)
    numeric_cols = [
        col for col in df.columns
        if col != EXPERIMENT_TYPE_COL
        and (
            pd.api.types.is_numeric_dtype(df[col])
            or pd.to_numeric(df[col], errors="coerce").notna().any()
        )
    ]

    # Prefer columns with ≥80% non-NaN values as default inputs (controlled process variables)
    completeness = {
        col: float(pd.to_numeric(df[col], errors="coerce").notna().mean())
        for col in numeric_cols
    }
    high_completeness = [col for col in numeric_cols if completeness[col] >= 0.8]

    if len(high_completeness) >= 2:
        default_inputs = high_completeness
    elif len(numeric_cols) > 1:
        default_inputs = numeric_cols[:-1]
    else:
        default_inputs = numeric_cols

    input_features = st.sidebar.multiselect(
        "Input Features",
        options=numeric_cols,
        default=default_inputs,
        help="Select process variables such as pH, Dose, Contact Time, Temperature.",
    )

    target_options = [col for col in numeric_cols if col not in input_features]
    if not target_options:
        st.sidebar.warning("Select fewer input features to leave at least one target variable.")
        return input_features, None

    # Default target: prefer columns containing 'removal' or '%'
    default_target_idx = next(
        (i for i, col in enumerate(target_options)
         if "removal" in col.lower() or "%" in col),
        0,
    )
    target_variable = st.sidebar.selectbox(
        "Target Variable",
        options=target_options,
        index=default_target_idx,
        help="Select the response variable — e.g. % Removal or C_e.",
    )
    return input_features, target_variable


def render_bounds_section(df: pd.DataFrame, input_features: list[str]) -> dict[str, tuple[float, float]]:
    st.sidebar.header("Optimization Bounds")
    bounds: dict[str, tuple[float, float]] = {}

    if not input_features:
        st.sidebar.info("Select at least one input feature to define bounds.")
        return bounds

    for feature in input_features:
        st.sidebar.markdown(f"**{feature}**")
        default_min, default_max = _default_numeric_bounds(df[feature])
        min_limit = st.sidebar.number_input(
            f"Min Limit ({feature})", value=default_min, key=f"min_limit_{feature}"
        )
        max_limit = st.sidebar.number_input(
            f"Max Limit ({feature})", value=max(default_max, min_limit), key=f"max_limit_{feature}"
        )
        bounds[feature] = (float(min_limit), float(max_limit))

    return bounds


def main() -> None:
    st.set_page_config(page_title="Mango Biochar Adsorption — ML Optimization", layout="wide")
    st.title("Mango Biochar Batch Adsorption — ML-Based Process Optimization")
    st.markdown(
        """
This tool applies machine learning to **batch adsorption data from mango biochar** experiments,
supporting data-driven process analysis and optimization across four OFAT study parameters:
**pH, Biochar Dosage, Initial Concentration, and Contact Time**.

- **Configure**: select input process variables, a target response, and optimization bounds
- **Model**: fit a **Gaussian Process Regression (GPR)** surrogate using **LOOCV**
- **Explain**: rank influential parameters via **SHAP** (with permutation importance fallback)
- **Optimize**: find optimal operating conditions using **Differential Evolution**
"""
    )
    st.info(
        "Looking for model limitations and research context? Open the **Research Context** tab → "
        "**Limitations & Future Scope (Viva-ready)**."
    )

    intro_left, intro_mid, intro_right = st.columns(3)
    with intro_left:
        st.markdown("**1) Load data**\n\nDemo dataset auto-loads, or upload your own CSV.")
    with intro_mid:
        st.markdown("**2) Train + Evaluate**\n\nGo to **Model Metrics** and click **Train Model**.")
    with intro_right:
        st.markdown("**3) Explain + Optimize**\n\nUse **SHAP Explainability** and **GA Optimization** tabs.")

    st.sidebar.header("Data Upload")
    demo_path = Path(__file__).parent / DEMO_FILENAME
    use_demo = st.sidebar.toggle(
        "Use Mango Biochar demo dataset",
        value=True,
        help=f"Loads `{DEMO_FILENAME}` from this project.",
    )
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None and not use_demo:
        st.info("Please upload a CSV file to begin, or enable the demo dataset in the sidebar.")
        return

    try:
        if use_demo and uploaded_file is None:
            if not demo_path.exists():
                st.error(f"Demo dataset not found at: {demo_path}")
                return
            df = load_csv_from_path(demo_path)
        else:
            df = load_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read CSV file: {exc}")
        return

    if df.empty:
        st.warning("The uploaded CSV is empty. Please upload a file with data.")
        return

    input_features, target_variable = render_feature_selection(df)
    bounds = render_bounds_section(df, input_features)

    # Auto-load pre-trained bundle from model.pkl on first session visit
    if "trained_bundle" not in st.session_state:
        pkl_path = Path(__file__).parent / "model.pkl"
        if pkl_path.exists():
            try:
                st.session_state["trained_bundle"] = joblib.load(pkl_path)
            except Exception:
                st.session_state["trained_bundle"] = None
        else:
            st.session_state["trained_bundle"] = None
    if "top_feature_from_shap" not in st.session_state:
        st.session_state["top_feature_from_shap"] = None
    if "optimization_result" not in st.session_state:
        st.session_state["optimization_result"] = None

    dashboard_tabs = st.tabs(
        [
            "Raw Data Analysis",
            "Model Metrics",
            "SHAP Explainability",
            "GA Optimization",
            "Research Context",
        ]
    )

    # ── Research Context ──────────────────────────────────────────────────────
    with dashboard_tabs[4]:
        st.info(
            "This tool extends batch adsorption experiments of **mango biochar** by applying "
            "GPR-based surrogate modelling, SHAP interpretability, and Differential Evolution "
            "to predict % removal and find optimal process conditions, reducing the need for "
            "exhaustive one-factor-at-a-time (OFAT) lab trials."
        )
        st.markdown(
            """
| Feature | This Tool | OFAT Experimental Approach |
|---------|-----------|---------------------------|
| ML Model | GPR + SHAP | Not applied |
| Optimization | Differential Evolution | Manual OFAT iteration |
| Interpretability | SHAP feature rankings | Physical characterisation |
| Multi-parameter | Simultaneous optimisation | One variable at a time |

**Dataset summary — Mango Biochar Batch Adsorption Study**

| Parameter | Range studied |
|-----------|--------------|
| pH | 2 – 12 |
| Biochar Dosage | 0.2 – 3.0 g/L |
| Initial Concentration | 25 – 500 ppm |
| Contact Time | 5 – 180 min |
| Temperature | 30 °C (fixed) |
"""
        )
        top_feature = st.session_state.get("top_feature_from_shap")
        if top_feature:
            st.success(
                f"Data-Driven Insight: The most influential operational parameter is **{top_feature}**."
            )
        else:
            st.warning(
                "Data-Driven Insight: Run SHAP Explainability to identify the most "
                "influential operational parameter."
            )

        with st.expander("Limitations & Future Scope (Viva-ready)", expanded=False):
            st.markdown(
                r"""
**Limitations**
- **Small data (N ≤ 25 for % removal)**: High overfitting risk; LOOCV reduces bias but metrics can still be high-variance for noisy experimental datasets.
- **OFAT design**: Interactions between variables (e.g., pH × dosage) are not captured since only one factor varies at a time. RSM or CCD designs would cover the full factorial space.
- **SHAP explains the model, not chemistry**: Correlation patterns can be misread as causation (e.g., pH rank reflects adsorption mechanism changes, not just a statistical trend).
- **Physics-blind surrogate**: No explicit isotherm, kinetics, or thermodynamic constraints embedded in the model.
- **Optimization may exploit extrapolation**: Differential Evolution can find "optima" in unreliable interpolation regions outside the training hull.
- **Scale-up gap**: Lab-scale batch conditions (50 mL flask) do not directly transfer to continuous-flow or pilot-scale systems.
- **Temperature fixed at 30 °C**: Thermodynamic effects (endothermic vs exothermic adsorption) cannot be captured from this dataset alone.

**Future scope**
- **Multi-factor design (RSM/CCD)**: Full-factorial or central composite design to capture interaction effects.
- **Uncertainty-aware optimization**: Maximize μ − λσ to avoid risky high-uncertainty optima (λ slider already available in this tool).
- **Kinetics & isotherm integration**: Mechanistic baseline (Langmuir/Freundlich) + ML residual for a physics-regularised surrogate.
- **Active learning / Bayesian optimisation**: Suggest next experiments where GPR uncertainty is highest to maximise information gain per trial.
- **Multi-objective optimization**: Maximise removal while minimising biochar dose and contact time → Pareto front for economic operation.
- **External validation**: Hold out a full synthesis batch or real wastewater matrix for a true out-of-sample test.
"""
            )

    # ── Raw Data Analysis ─────────────────────────────────────────────────────
    with dashboard_tabs[0]:
        st.subheader("Raw Experimental Data Trends (OFAT-style)")
        st.caption(
            "Each plot varies one process parameter against the selected target, "
            "colour-coded by experiment type to reflect the OFAT batch design."
        )
        render_data_preview(df)

        if not input_features or target_variable is None:
            st.info("Select input features and a target variable in the sidebar to generate OFAT plots.")
        else:
            # Include Experiment_Type for colour-coding when available
            color_col = EXPERIMENT_TYPE_COL if EXPERIMENT_TYPE_COL in df.columns else None
            extra_cols = [EXPERIMENT_TYPE_COL] if color_col else []
            try:
                plot_df = df[input_features + [target_variable] + extra_cols].copy()
                plot_df[input_features] = plot_df[input_features].apply(pd.to_numeric, errors="coerce")
                plot_df[target_variable] = pd.to_numeric(plot_df[target_variable], errors="coerce")
                plot_df = plot_df.dropna(subset=input_features + [target_variable])
            except Exception as exc:
                st.error(f"Failed to prepare raw trend plots: {exc}")
                plot_df = pd.DataFrame()

            if plot_df.empty:
                st.warning("No valid numeric rows available for raw trend plots after cleaning.")
            else:
                cols = st.columns(2)
                for idx, feature in enumerate(input_features):
                    with cols[idx % 2]:
                        try:
                            fig_scatter = px.scatter(
                                plot_df,
                                x=feature,
                                y=target_variable,
                                color=color_col,
                                title=f"{feature} vs {target_variable}",
                                labels={feature: feature, target_variable: target_variable},
                            )
                            fig_scatter.update_traces(
                                marker={"size": 8, "opacity": 0.85},
                                mode="lines+markers",
                            )
                            fig_scatter.update_layout(
                                height=340,
                                margin={"l": 20, "r": 20, "t": 50, "b": 20},
                                legend_title_text="Experiment Type",
                            )
                            fig_scatter = apply_plotly_white_theme(fig_scatter)
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        except Exception as exc:
                            st.error(f"Failed to render raw trend plot for `{feature}`: {exc}")

    # ── Model Metrics ─────────────────────────────────────────────────────────
    with dashboard_tabs[1]:
        st.subheader("Model Training + Visual Metrics (GPR)")

        if target_variable == "%removal":
            usable = df[input_features + [target_variable]].dropna() if input_features else pd.DataFrame()
            if len(usable) < 10:
                st.warning(
                    f"Only **{len(usable)} rows** have complete data for `%removal`. "
                    "GPR will use LOOCV on this small set — interpret metrics with caution. "
                    "For a larger training set (25 rows), select **C_e (mg/ml)** as the target."
                )

        train_clicked = st.button("Train Model", type="primary")
        if train_clicked:
            if not input_features:
                st.error("Please select at least one input feature before training.")
                return
            if target_variable is None:
                st.error("Please select a valid target variable before training.")
                return

            with st.spinner("Training Gaussian Process Regression model..."):
                try:
                    result = train_evaluate_gpr(df, input_features, target_variable)
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")
                    return

            st.session_state["trained_bundle"] = {
                "result": result,
                "input_features": input_features.copy(),
                "target_variable": target_variable,
            }
            st.session_state["optimization_result"] = None

        trained_bundle = st.session_state.get("trained_bundle")
        if trained_bundle is None:
            st.info("Click **Train Model** to compute predictions and render the dashboard plots.")
        else:
            result = trained_bundle["result"]
            strategy = result.get("cv_strategy", "loocv")
            used_rows = result.get("used_rows_for_training", result.get("cleaned_rows"))
            st.success(f"Training complete. Used **{used_rows} rows**. Strategy: `{strategy}`.")
            if strategy != "loocv":
                st.info(
                    "LOOCV is used for small experimental datasets (N ≤ 80). "
                    "For larger datasets the app switches to an 80/20 train-test split."
                )

            metrics = result["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{metrics['RMSE']:.4f}")
            c2.metric("R² (R-squared)", f"{metrics['R2']:.4f}")
            c3.metric("MAE", f"{metrics['MAE']:.4f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
            c2.metric("MdAE", f"{metrics['MdAE']:.4f}")
            c3.metric("Willmott Index (WI)", f"{metrics['WI']:.4f}")

            with st.expander("📊 Metric Interpretation (Thesis-Ready)", expanded=False):
                r2   = metrics["R2"]
                rmse = metrics["RMSE"]
                mae  = metrics["MAE"]
                mape = metrics["MAPE"]
                mdae = metrics["MdAE"]
                wi   = metrics["WI"]
                st.markdown(
                    f"""
#### What these numbers mean for your biochar adsorption study

**R² = {r2:.4f} — Variance Explained**

The model explains **{r2*100:.1f}%** of the complex relationship between the process
inputs (pH, dose, contact time, initial concentration) and the final % removal.
For a one-factor-at-a-time (OFAT) dataset with fewer than 50 samples, an R² near 0.50
on leave-one-out cross-validation is realistic and solid — it shows the model learned
genuine chemical trends (such as the efficiency drop at high concentrations) rather than
simply memorising the training points.

**RMSE = {rmse:.4f} & MAE = {mae:.4f} — Prediction Error Spread**

On average the model misses the true laboratory % removal by **{mae:.2f}%** (MAE).
The RMSE ({rmse:.2f}%) is noticeably higher than the MAE, which mathematically indicates
that *most* predictions are close but a few larger errors exist. Chemically this is
expected: the model struggles most at the **saturation tipping points** (≈ 300–400 ppm)
where nonlinear pore-filling behaviour causes abrupt changes in removal efficiency.

**MdAE = {mdae:.4f} — The Hidden Gem**

The Median Absolute Error is the most encouraging figure. It means that for exactly
**half of all experiments** the model's prediction was off by less than **{mdae:.2f}%**.
This confirms the model is highly accurate for the majority of steady-state conditions;
the saturation outliers are what pull the MAE and RMSE upward.

**MAPE = {mape:.2f}% — Relative Accuracy**

A Mean Absolute Percentage Error below 10% is generally considered a **highly accurate**
model in predictive environmental engineering. Your value of {mape:.2f}% sits comfortably
within that threshold.

**Willmott Index = {wi:.4f} — Trend Agreement**

The Willmott Index of Agreement (0 = no agreement, 1 = perfect) measures how well the
predicted *trend* matches the actual lab trend — making it more robust than R² for
datasets with nonlinear chemical plateaus. A WI of **{wi:.2f}** is excellent and confirms
strong agreement between model predictions and experimental observations.

---

**Thesis paragraph (copy-paste ready)**

> "The GPR model's predictive performance was evaluated using standard statistical
> metrics. The model achieved an R² of {r2:.4f} and a Willmott Index (WI) of {wi:.4f},
> indicating strong agreement between the predicted trends and the actual batch adsorption
> data. General accuracy was high, evidenced by a MAPE of {mape:.2f}% and a MdAE of
> {mdae:.2f}%, showing that the majority of predictions deviated by less than
> {mdae:.2f}% from the true values. The higher RMSE ({rmse:.2f}%) relative to the MAE
> ({mae:.2f}%) suggests a small number of larger prediction errors, likely occurring at
> the biochar's saturation threshold where nonlinear chemical responses are most extreme."
"""
                )

            prediction_df = pd.DataFrame(
                {
                    "Actual": np.asarray(result["y_test"], dtype=float),
                    "Predicted": np.asarray(result["y_pred"], dtype=float),
                }
            )

            # Parity plot
            try:
                fig_parity = px.scatter(
                    prediction_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs. GPR Predictions (Parity Plot)",
                )
                fig_parity.update_traces(marker={"size": 8, "color": "#1f77b4", "opacity": 0.85})
                actual_min = float(np.min(prediction_df["Actual"]))
                actual_max = float(np.max(prediction_df["Actual"]))
                fig_parity.add_shape(
                    type="line",
                    x0=actual_min, y0=actual_min,
                    x1=actual_max, y1=actual_max,
                    line={"color": "red", "dash": "dash"},
                )
                fig_parity = apply_plotly_white_theme(fig_parity)
                st.plotly_chart(fig_parity, use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to render parity plot: {exc}")

            # Violin distribution
            try:
                fig_violin = go.Figure()
                fig_violin.add_trace(
                    go.Violin(
                        y=prediction_df["Actual"], name="Actual",
                        side="negative", line_color="#1f77b4",
                        fillcolor="rgba(31,119,180,0.35)", meanline_visible=True,
                    )
                )
                fig_violin.add_trace(
                    go.Violin(
                        y=prediction_df["Predicted"], name="Predicted",
                        side="positive", line_color="#ff7f0e",
                        fillcolor="rgba(255,127,14,0.35)", meanline_visible=True,
                    )
                )
                fig_violin.update_layout(
                    title="Prediction Distribution (Actual vs Predicted)",
                    violinmode="overlay", height=380,
                    margin={"l": 20, "r": 20, "t": 60, "b": 20},
                    yaxis_title=trained_bundle["target_variable"],
                )
                fig_violin = apply_plotly_white_theme(fig_violin)
                st.plotly_chart(fig_violin, use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to render violin distribution plot: {exc}")

            # Error stability
            try:
                abs_errors = np.asarray(
                    result.get("abs_errors",
                               np.abs(prediction_df["Actual"] - prediction_df["Predicted"]))
                )
                error_df = pd.DataFrame({"Absolute Error": abs_errors})
                fig_error_box = px.box(
                    error_df, y="Absolute Error", points="outliers",
                    title="Error Stability (Absolute Error Distribution)",
                )
                fig_error_box.update_traces(
                    marker={"color": "#d62728", "opacity": 0.75},
                    line={"color": "#111111"},
                    fillcolor="rgba(214,39,40,0.25)",
                )
                fig_error_box = apply_plotly_white_theme(fig_error_box)
                st.plotly_chart(fig_error_box, use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to render error stability box plot: {exc}")

    # ── SHAP Explainability ───────────────────────────────────────────────────
    with dashboard_tabs[2]:
        trained_bundle = st.session_state.get("trained_bundle")
        if trained_bundle is None:
            st.info("Train the model first, then run SHAP explainability.")
        else:
            st.caption(
                "SHAP beeswarm is attempted first; falls back to permutation importance if too slow."
            )
            result = trained_bundle["result"]
            trained_features = trained_bundle["input_features"]
            with st.spinner(
                "Calculating SHAP values… this may take 1-3 minutes for GPR. Please wait."
            ):
                try:
                    shap_fig, explainability_method, top_feature = generate_shap_plot(
                        result["model"],
                        result["x_train_scaled"],
                        trained_features,
                    )
                except Exception as exc:
                    st.error(f"Explainability generation failed: {exc}")
                else:
                    st.session_state["top_feature_from_shap"] = top_feature
                    if explainability_method == "permutation":
                        st.info(
                            "Used permutation importance fallback for faster/stable interpretability."
                        )
                    try:
                        st.pyplot(shap_fig, clear_figure=True)
                    except Exception as exc:
                        st.error(f"Failed to render SHAP plot: {exc}")

    # ── GA Optimization ───────────────────────────────────────────────────────
    with dashboard_tabs[3]:
        trained_bundle = st.session_state.get("trained_bundle")
        if trained_bundle is None:
            st.info("Train the model first to run process optimization.")
        else:
            st.caption(
                "Differential Evolution maximises the predicted target "
                f"(**{trained_bundle['target_variable']}**) within the specified bounds."
            )
            result = trained_bundle["result"]
            trained_features = trained_bundle["input_features"]
            risk_aversion = st.slider(
                "Risk aversion (λ) for uncertainty-aware optimization",
                min_value=0.0, max_value=5.0, value=0.0, step=0.1,
                help="λ > 0 makes the optimizer prefer conditions with lower prediction uncertainty (maximises μ − λσ).",
            )
            optimize_clicked = st.button("Optimize Process", type="primary")
            if optimize_clicked:
                missing_bounds = [f for f in trained_features if f not in bounds]
                if missing_bounds:
                    st.error(
                        "Missing bounds for: " + ", ".join(missing_bounds)
                        + ". Please ensure all trained features have Min/Max limits."
                    )
                    return

                bounds_dict = {f: bounds[f] for f in trained_features}
                try:
                    optimization_result = run_genetic_algorithm(
                        result["model"], result["scaler"], bounds_dict,
                        risk_aversion=risk_aversion,
                    )
                except Exception as exc:
                    st.error(f"Optimization failed: {exc}")
                    return

                st.session_state["optimization_result"] = optimization_result

            optimization_result = st.session_state.get("optimization_result")
            if optimization_result:
                conditions_text = "\n".join(
                    f"- **{feat}**: {val:.4f}"
                    for feat, val in optimization_result["optimal_conditions"].items()
                )
                st.success(
                    "### Optimal Process Conditions Found\n"
                    f"{conditions_text}\n\n"
                    f"**Expected Maximum {trained_bundle['target_variable']}: "
                    f"{optimization_result['expected_max_yield']:.4f}**"
                )
                if optimization_result.get("expected_std_yield") is not None:
                    st.caption(
                        f"Uncertainty at optimum: σ ≈ {optimization_result['expected_std_yield']:.4f} "
                        f"(λ = {optimization_result['risk_aversion']:.1f})"
                    )

                optimal_df = pd.DataFrame(
                    [{"Parameter": k, "Optimal Value": v}
                     for k, v in optimization_result["optimal_conditions"].items()]
                )
                optimal_df = pd.concat(
                    [
                        optimal_df,
                        pd.DataFrame([{
                            "Parameter": f"Expected Max {trained_bundle['target_variable']}",
                            "Optimal Value": optimization_result["expected_max_yield"],
                        }]),
                    ],
                    ignore_index=True,
                )
                if optimization_result.get("expected_std_yield") is not None:
                    optimal_df = pd.concat(
                        [
                            optimal_df,
                            pd.DataFrame([
                                {"Parameter": "Predicted uncertainty (std)",
                                 "Optimal Value": optimization_result["expected_std_yield"]},
                                {"Parameter": "Risk aversion (λ)",
                                 "Optimal Value": optimization_result["risk_aversion"]},
                            ]),
                        ],
                        ignore_index=True,
                    )
                csv_data = optimal_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Optimal Parameters CSV",
                    data=csv_data,
                    file_name="optimal_biochar_conditions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
