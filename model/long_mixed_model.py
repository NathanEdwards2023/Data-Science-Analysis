from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


project_dir = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_dir / "brewery_data_cleaned.csv")

loss_columns = {
    "Loss_During_Brewing": "Brewing",
    "Loss_During_Fermentation": "Fermentation",
    "Loss_During_Bottling_Kegging": "Bottling/Kegging",
}

# Convert the three wide loss columns into one long outcome column.
long_data = data.melt(
    id_vars=[
        "Batch_ID", "Beer_Style", "SKU", "Location", "Volume_Produced",
        "Fermentation_Time", "Temperature",
    ],
    value_vars=list(loss_columns),
    var_name="Loss_Stage",
    value_name="Loss_Value",
).assign(Loss_Stage=lambda frame: frame["Loss_Stage"].map(loss_columns))
long_data["Loss_Stage"] = pd.Categorical(
    long_data["Loss_Stage"],
    categories=["Brewing", "Fermentation", "Bottling/Kegging"],
    ordered=True,
)
long_data.to_csv(project_dir / "production_loss_long.csv", index=False)

formula = (
    "Loss_Value ~ C(Loss_Stage, Treatment(reference='Brewing')) + "
    "C(Beer_Style, Treatment(reference='Lager')) + "
    "C(SKU, Treatment(reference='Bottles')) + "
    "C(Location, Treatment(reference='Indiranagar')) + "
    "Volume_Produced + Fermentation_Time + Temperature"
)

mixed_model = smf.mixedlm(
    formula=formula,
    data=long_data,
    groups=long_data["Batch_ID"],
    re_formula="1",
).fit(reml=False, method="lbfgs")

stage_terms = [
    "C(Loss_Stage, Treatment(reference='Brewing'))[T.Fermentation]",
    "C(Loss_Stage, Treatment(reference='Brewing'))[T.Bottling/Kegging]",
]
stage_names = mixed_model.model.exog_names
stage_indices = [stage_names.index(term) for term in stage_terms]
stage_wald = mixed_model.wald_test_terms(skip_single=False)
stage_table = stage_wald.table
stage_test = stage_table.loc["C(Loss_Stage, Treatment(reference='Brewing'))"]
stage_statistic = float(stage_test["statistic"])
stage_p_value = float(stage_test["pvalue"])
stage_df = int(stage_test["df_constraint"])

stage_means = (
    long_data.groupby("Loss_Stage", observed=False)["Loss_Value"]
    .agg(Mean="mean", SD="std", N="count")
    .reset_index()
)

source_labels = {
    "C(Loss_Stage, Treatment(reference='Brewing'))": "Loss stage",
    "C(Beer_Style, Treatment(reference='Lager'))": "Beer style",
    "C(SKU, Treatment(reference='Bottles'))": "SKU",
    "C(Location, Treatment(reference='Indiranagar'))": "Location",
    "Volume_Produced": "Production volume",
    "Fermentation_Time": "Fermentation time",
    "Temperature": "Temperature",
}
apa_table = pd.DataFrame({
    "Source": [source_labels.get(term, term) for term in stage_table.index],
    "df": stage_table["df_constraint"].astype(int).to_numpy(),
    "Wald chi-square": stage_table["statistic"].astype(float).to_numpy(),
    "p": stage_table["pvalue"].astype(float).to_numpy(),
})

report = [
    "LONG-FORMAT MIXED-EFFECTS MODEL",
    f"Observations: {len(long_data):,}",
    f"Batches/random-intercept groups: {long_data['Batch_ID'].nunique():,}",
    f"Formula: {formula}",
    "Random effect: Batch_ID random intercept",
    "",
    "LOSS-STAGE MEANS",
    stage_means.round(4).to_string(index=False),
    "",
    "LOSS-STAGE JOINT WALD TEST",
    f"Chi-square({stage_df}) = {stage_statistic:.4f}, p = {stage_p_value:.6g}",
    "",
    "APA-STYLE ANOVA TABLE",
    apa_table.round(4).to_string(index=False),
]
(project_dir / "model" / "long_mixed_model_report.txt").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)
stage_means.to_csv(project_dir / "model" / "loss_stage_means.csv", index=False)
apa_table.to_csv(project_dir / "model" / "long_mixed_model_anova.csv", index=False)
print("\n".join(report))
