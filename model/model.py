from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


project_dir = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_dir / "brewery_data_cleaned.csv")

# Model A: fermentation loss by categorical production descriptors and process measures.
# C(...) specifies nominal predictors; statsmodels uses treatment coding by default.
formula = (
    "Loss_During_Fermentation ~ "
    "C(Beer_Style, Treatment(reference='Lager')) + "
    "C(SKU, Treatment(reference='Bottles')) + "
    "C(Location, Treatment(reference='Indiranagar')) + "
    "Volume_Produced + Fermentation_Time + Temperature"
)
model_a = smf.ols(formula=formula, data=data).fit()

coefficient_table = model_a.summary2().tables[1].reset_index()
coefficient_table = coefficient_table.rename(columns={"index": "term"})
coefficient_table.to_csv(project_dir / "model" / "model_a_coefficients.csv", index=False)

# Model diagnostics
residuals = model_a.resid
shapiro_w, shapiro_p = shapiro(residuals)
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, model_a.model.exog)
dw = durbin_watson(residuals)

design_matrix = pd.DataFrame(
    model_a.model.exog,
    columns=model_a.model.exog_names,
)
vif_table = pd.DataFrame({
    "term": design_matrix.columns,
    "VIF": [
        variance_inflation_factor(design_matrix.values, i)
        for i in range(design_matrix.shape[1])
    ],
})
vif_table.to_csv(project_dir / "model" / "model_a_vif.csv", index=False)

influence = model_a.get_influence()
cooks_distance = influence.cooks_distance[0]
max_cook_index = int(cooks_distance.argmax())
max_cook = float(cooks_distance[max_cook_index])
cook_threshold = 4 / len(data)
influential_count = int((cooks_distance > cook_threshold).sum())

report = [
    "MODEL A: FERMENTATION LOSS",
    f"Formula: {formula}",
    f"Observations: {int(model_a.nobs):,}",
    f"R-squared: {model_a.rsquared:.6f}",
    f"Adjusted R-squared: {model_a.rsquared_adj:.6f}",
    f"F-statistic: {model_a.fvalue:.6f}",
    f"F-test p-value: {model_a.f_pvalue:.6g}",
    f"Residual standard error: {model_a.mse_resid ** 0.5:.6f}",
    "",
    "COEFFICIENTS",
    coefficient_table.to_string(index=False),
    "",
    "REFERENCE LEVELS",
    "Beer_Style: Lager; SKU: Bottles; Location: Indiranagar",
    "",
    "MODEL DIAGNOSTICS",
    f"Normality, Shapiro-Wilk W: {shapiro_w:.6f}; p-value: {shapiro_p:.6g}",
    f"Homoscedasticity, Breusch-Pagan LM: {bp_lm:.6f}; LM p-value: {bp_lm_p:.6g}; F: {bp_f:.6f}; F p-value: {bp_f_p:.6g}",
    f"Independence, Durbin-Watson statistic: {dw:.6f}",
    f"Multicollinearity, maximum VIF excluding intercept: {vif_table.loc[vif_table.term != 'Intercept', 'VIF'].max():.6f}",
    vif_table.to_string(index=False),
    f"Influential observations, Cook's distance threshold (4/n): {cook_threshold:.6f}",
    f"Count above threshold: {influential_count}",
    f"Maximum Cook's distance: {max_cook:.6f} at row index {max_cook_index}",
]

report_text = "\n".join(report) + "\n"
(project_dir / "model" / "model_a_report.txt").write_text(report_text, encoding="utf-8")
print(report_text)


# Model B: loss during brewing, using the same reference categories as Model A.
model_b_formula = (
    "Loss_During_Brewing ~ "
    "C(Beer_Style, Treatment(reference='Lager')) + "
    "C(SKU, Treatment(reference='Bottles')) + "
    "C(Location, Treatment(reference='Indiranagar')) + "
    "Volume_Produced + Fermentation_Time + Temperature"
)
model_b = smf.ols(formula=model_b_formula, data=data).fit()
model_b_coefficients = model_b.summary2().tables[1].reset_index()
model_b_coefficients = model_b_coefficients.rename(columns={"index": "term"})
model_b_coefficients.to_csv(
    project_dir / "model" / "model_b_coefficients.csv", index=False
)

model_b_report = [
    "MODEL B: BREWING LOSS",
    f"Formula: {model_b_formula}",
    f"Observations: {int(model_b.nobs):,}",
    f"R-squared: {model_b.rsquared:.6f}",
    f"Adjusted R-squared: {model_b.rsquared_adj:.6f}",
    f"F-statistic: {model_b.fvalue:.6f}",
    f"F-test p-value: {model_b.f_pvalue:.6g}",
    f"Residual standard error: {model_b.mse_resid ** 0.5:.6f}",
    "",
    "REFERENCE LEVELS",
    "Beer_Style: Lager; SKU: Bottles; Location: Indiranagar",
    "",
    "COEFFICIENTS",
    model_b_coefficients.to_string(index=False),
]
model_b_report_text = "\n".join(model_b_report) + "\n"
(project_dir / "model" / "model_b_report.txt").write_text(
    model_b_report_text, encoding="utf-8"
)
print(model_b_report_text)


# Model C: loss during bottling/kegging, using the same reference categories.
model_c_formula = (
    "Loss_During_Bottling_Kegging ~ "
    "C(Beer_Style, Treatment(reference='Lager')) + "
    "C(SKU, Treatment(reference='Bottles')) + "
    "C(Location, Treatment(reference='Indiranagar')) + "
    "Volume_Produced + Fermentation_Time + Temperature"
)
model_c = smf.ols(formula=model_c_formula, data=data).fit()
model_c_coefficients = model_c.summary2().tables[1].reset_index()
model_c_coefficients = model_c_coefficients.rename(columns={"index": "term"})
model_c_coefficients.to_csv(
    project_dir / "model" / "model_c_coefficients.csv", index=False
)

model_c_report = [
    "MODEL C: BOTTLING/KEGGING LOSS",
    f"Formula: {model_c_formula}",
    f"Observations: {int(model_c.nobs):,}",
    f"R-squared: {model_c.rsquared:.6f}",
    f"Adjusted R-squared: {model_c.rsquared_adj:.6f}",
    f"F-statistic: {model_c.fvalue:.6f}",
    f"F-test p-value: {model_c.f_pvalue:.6g}",
    f"Residual standard error: {model_c.mse_resid ** 0.5:.6f}",
    "",
    "REFERENCE LEVELS",
    "Beer_Style: Lager; SKU: Bottles; Location: Indiranagar",
    "",
    "COEFFICIENTS",
    model_c_coefficients.to_string(index=False),
]
model_c_report_text = "\n".join(model_c_report) + "\n"
(project_dir / "model" / "model_c_report.txt").write_text(
    model_c_report_text, encoding="utf-8"
)
print(model_c_report_text)


def write_diagnostic_report(model, model_name, output_stem):
    residuals = model.resid
    shapiro_w, shapiro_p = shapiro(residuals)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
        residuals, model.model.exog
    )
    dw = durbin_watson(residuals)
    design_matrix = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    vif_table = pd.DataFrame({
        "term": design_matrix.columns,
        "VIF": [
            variance_inflation_factor(design_matrix.values, i)
            for i in range(design_matrix.shape[1])
        ],
    })
    vif_table.to_csv(project_dir / "model" / f"{output_stem}_vif.csv", index=False)

    cooks_distance = model.get_influence().cooks_distance[0]
    cook_threshold = 4 / len(data)
    max_cook_index = int(cooks_distance.argmax())

    diagnostic_report = [
        f"{model_name} DIAGNOSTICS",
        f"Observations: {int(model.nobs):,}",
        f"Normality, Shapiro-Wilk W: {shapiro_w:.6f}; p-value: {shapiro_p:.6g}",
        f"Homoscedasticity, Breusch-Pagan LM: {bp_lm:.6f}; LM p-value: {bp_lm_p:.6g}; F: {bp_f:.6f}; F p-value: {bp_f_p:.6g}",
        f"Independence, Durbin-Watson statistic: {dw:.6f}",
        f"Multicollinearity, maximum VIF excluding intercept: {vif_table.loc[vif_table.term != 'Intercept', 'VIF'].max():.6f}",
        vif_table.to_string(index=False),
        f"Influential observations, Cook's distance threshold (4/n): {cook_threshold:.6f}",
        f"Count above threshold: {int((cooks_distance > cook_threshold).sum())}",
        f"Maximum Cook's distance: {cooks_distance[max_cook_index]:.6f} at row index {max_cook_index}",
    ]
    report = "\n".join(diagnostic_report) + "\n"
    (project_dir / "model" / f"{output_stem}_diagnostics.txt").write_text(
        report, encoding="utf-8"
    )
    print(report)


write_diagnostic_report(model_b, "MODEL B: BREWING LOSS", "model_b")
write_diagnostic_report(model_c, "MODEL C: BOTTLING/KEGGING LOSS", "model_c")


# Information-criterion comparison across the three fitted models.
model_comparison = pd.DataFrame({
    "model": ["Model A", "Model B", "Model C"],
    "dependent_variable": [
        "Loss_During_Fermentation",
        "Loss_During_Brewing",
        "Loss_During_Bottling_Kegging",
    ],
    "AIC": [model_a.aic, model_b.aic, model_c.aic],
    "BIC": [model_a.bic, model_b.bic, model_c.bic],
    "R_squared": [model_a.rsquared, model_b.rsquared, model_c.rsquared],
})
model_comparison.to_csv(
    project_dir / "model" / "model_comparison.csv", index=False
)
print("MODEL COMPARISON\n" + model_comparison.round(6).to_string(index=False))


# Estimated marginal means (balanced over categorical predictors; numeric
# predictors held at their sample means).
categorical_predictors = ["Beer_Style", "SKU", "Location"]
numeric_predictors = ["Volume_Produced", "Fermentation_Time", "Temperature"]
categorical_levels = {
    column: sorted(data[column].dropna().unique())
    for column in categorical_predictors
}
mean_numeric = data[numeric_predictors].mean().to_dict()

emm_rows = []
for model_name, fitted_model in {
    "Model A": model_a,
    "Model B": model_b,
    "Model C": model_c,
}.items():
    all_combinations = pd.MultiIndex.from_product(
        [categorical_levels[column] for column in categorical_predictors],
        names=categorical_predictors,
    ).to_frame(index=False)
    prediction_grid = all_combinations.assign(**mean_numeric)
    prediction_grid["prediction"] = fitted_model.predict(prediction_grid)
    for factor in categorical_predictors:
        factor_means = (
            prediction_grid.groupby(factor, as_index=False)["prediction"]
            .mean()
            .rename(columns={"prediction": "estimated_marginal_mean"})
        )
        factor_means.insert(0, "model", model_name)
        factor_means.insert(1, "factor", factor)
        emm_rows.append(factor_means)

estimated_marginal_means = pd.concat(emm_rows, ignore_index=True)
estimated_marginal_means.to_csv(
    project_dir / "model" / "estimated_marginal_means.csv", index=False
)
emm_report = [
    "ESTIMATED MARGINAL MEANS",
    "Categorical predictors averaged over all categorical-level combinations.",
    "Volume_Produced, Fermentation_Time, and Temperature held at sample means.",
    estimated_marginal_means.round(6).to_string(index=False),
]
(project_dir / "model" / "estimated_marginal_means.txt").write_text(
    "\n".join(emm_report) + "\n", encoding="utf-8"
)
print("ESTIMATED MARGINAL MEANS\n" + estimated_marginal_means.round(6).to_string(index=False))


# Crude models: one predictor at a time for each loss outcome.
outcomes = {
    "Model A": "Loss_During_Fermentation",
    "Model B": "Loss_During_Brewing",
    "Model C": "Loss_During_Bottling_Kegging",
}
predictor_formulas = {
    "Beer Style": "C(Beer_Style, Treatment(reference='Lager'))",
    "SKU": "C(SKU, Treatment(reference='Bottles'))",
    "Location": "C(Location, Treatment(reference='Indiranagar'))",
    "Volume Produced": "Volume_Produced",
    "Fermentation Time": "Fermentation_Time",
    "Temperature": "Temperature",
}

crude_rows = []
for model_name, outcome in outcomes.items():
    for predictor_name, predictor_formula in predictor_formulas.items():
        crude_formula = f"{outcome} ~ {predictor_formula}"
        crude_model = smf.ols(crude_formula, data=data).fit()
        table = crude_model.summary2().tables[1].reset_index()
        table = table.rename(columns={"index": "term"})
        for _, row in table.iterrows():
            crude_rows.append({
                "model": model_name,
                "dependent_variable": outcome,
                "predictor": predictor_name,
                "term": row["term"],
                "B": row["Coef."],
                "SE": row["Std.Err."],
                "t": row["t"],
                "p": row["P>|t|"],
                "R_squared": crude_model.rsquared,
                "Adjusted_R_squared": crude_model.rsquared_adj,
                "AIC": crude_model.aic,
                "N": int(crude_model.nobs),
            })

crude_summary = pd.DataFrame(crude_rows)
crude_summary.to_csv(
    project_dir / "model" / "crude_models_summary.csv", index=False
)
(project_dir / "model" / "crude_models_summary.txt").write_text(
    "CRUDE MODEL SUMMARY\n"
    "Each model contains one predictor and one loss outcome.\n\n"
    + crude_summary.round(6).to_string(index=False)
    + "\n",
    encoding="utf-8",
)

for model_name, outcome in outcomes.items():
    outcome_key = model_name.lower().replace(" ", "_")
    outcome_table = crude_summary[crude_summary["model"] == model_name].copy()
    outcome_table.to_csv(
        project_dir / "model" / f"{outcome_key}_crude_models.csv", index=False
    )
    (project_dir / "model" / f"{outcome_key}_crude_models.txt").write_text(
        f"{model_name}: CRUDE MODELS\n"
        f"Dependent variable: {outcome}\n\n"
        + outcome_table.round(6).to_string(index=False)
        + "\n",
        encoding="utf-8",
    )


# APA-style table for the three full multivariable models.
full_models = {"Model A": model_a, "Model B": model_b, "Model C": model_c}
apa_rows = []
for model_name, fitted_model in full_models.items():
    coefficient_table = fitted_model.summary2().tables[1].reset_index()
    coefficient_table = coefficient_table.rename(columns={"index": "term"})
    for _, row in coefficient_table.iterrows():
        apa_rows.append({
            "Model": model_name,
            "Term": row["term"],
            "B": row["Coef."],
            "SE": row["Std.Err."],
            "t": row["t"],
            "p": row["P>|t|"],
        })

apa_table = pd.DataFrame(apa_rows)
apa_table.to_csv(project_dir / "model" / "apa_models_abc_table.csv", index=False)
fit_table = pd.DataFrame({
    "Model": ["Model A", "Model B", "Model C"],
    "R²": [model_a.rsquared, model_b.rsquared, model_c.rsquared],
    "Adjusted R²": [model_a.rsquared_adj, model_b.rsquared_adj, model_c.rsquared_adj],
    "AIC": [model_a.aic, model_b.aic, model_c.aic],
    "F": [model_a.fvalue, model_b.fvalue, model_c.fvalue],
    "p": [model_a.f_pvalue, model_b.f_pvalue, model_c.f_pvalue],
    "N": [int(model_a.nobs), int(model_b.nobs), int(model_c.nobs)],
})
fit_table.to_csv(project_dir / "model" / "apa_models_abc_fit.csv", index=False)

apa_text = [
    "Table 1",
    "Ordinary Least Squares Regression Results for Production Loss Outcomes",
    "",
    "Model fit",
    fit_table.round(4).to_string(index=False),
    "",
    "Coefficients",
    apa_table.round(4).to_string(index=False),
    "",
    "Note. B = unstandardized coefficient; SE = standard error. Reference groups are Lager, Bottles, and Indiranagar.",
]
(project_dir / "model" / "apa_models_abc_table.txt").write_text(
    "\n".join(apa_text) + "\n", encoding="utf-8"
)
print("CRUDE MODEL SUMMARY\n" + crude_summary.round(4).to_string(index=False))
print("APA MODEL FIT TABLE\n" + fit_table.round(4).to_string(index=False))
