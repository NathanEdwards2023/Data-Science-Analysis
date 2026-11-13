from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from plotnine import (
    aes, element_blank, element_text, geom_boxplot, geom_col, geom_rect,
    geom_text, ggplot, labs, scale_fill_identity, scale_fill_manual,
    scale_y_continuous, theme, theme_minimal, theme_void, xlim, ylim,
)

project_dir = Path(__file__).resolve().parents[1]


# %% Initial dataset exploration
data = pd.read_csv(project_dir / "brewery_data.csv")

loss_columns = [
    "Loss_During_Brewing",
    "Loss_During_Fermentation",
    "Loss_During_Bottling_Kegging",
]
loss_summary = (
    data[loss_columns]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .T.sort_values("mean", ascending=False)
    .round(3)
)

report = [
    "BREWERY DATA: INITIAL EXPLORATION",
    f"Rows: {len(data):,}; columns: {data.shape[1]}",
    f"Unique batch IDs: {data['Batch_ID'].nunique():,}",
    f"Date range: {data['Brew_Date'].min()} to {data['Brew_Date'].max()}",
    f"Missing cells: {data.isna().sum().sum()}",
    f"Duplicate rows: {data.duplicated().sum()}",
    "\nCATEGORICAL COVERAGE",
]
for column in ["Beer_Style", "SKU", "Location"]:
    report.extend([
        f"\n{column}: {data[column].nunique()} categories",
        data[column].value_counts().to_string(),
    ])

report.extend([
    "\nFIELDS",
    ", ".join(data.columns),
    "\nLOSS BY STAGE (original recorded units; unweighted batch summaries)",
    loss_summary.to_string(),
    "\nRESEARCH QUESTION SUITABILITY",
    "The data supports an initial comparison of recorded loss across brewing, "
    "fermentation, and bottling/kegging. Fermentation has the highest mean "
    "recorded value, followed by bottling/kegging and brewing.",
    "This ranking assumes the loss columns use comparable units and definitions. "
    "The CSV supplies no data dictionary, loss units, or denominators. Do not "
    "treat the values as percentages or liters, sum them into total loss, or "
    "estimate lost volume without confirming these definitions.",
    "Beer style, packaging, location, date, and process measurements could "
    "support subsequent exploration of differences in recorded losses.",
    "The file identifies production stages, but does not explicitly record "
    "root causes such as spills, transfers, discarded batches, or equipment faults. "
    "It cannot establish causal sources of loss from these fields alone.",
    "The source, sampling method, measurement procedures, and whether these "
    "records represent actual craft brewing operations are not documented in "
    "the CSV. Generalization beyond these records requires that context.",
    "To answer which sources account for the most lost beer, obtain stage input "
    "and output volumes, consistent loss definitions, and loss reason codes. "
    "For financial loss, obtain corresponding cost information.",
])

report_text = "\n".join(report) + "\n"
(project_dir / "explore" / "production_loss_exploration.txt").write_text(
    report_text, encoding="utf-8"
)
print(report_text)


# %% Key variable units table
table = pd.DataFrame({
    "variable": [
        "Variable", "Brewing loss", "Fermentation loss", "Bottling/kegging loss",
        "Volume produced", "Brewhouse efficiency", "Fermentation time",
        "Fermentation temperature", "pH level", "Gravity", "Alcohol content",
    ],
    "unit": [
        "Unit / scale", "%", "%", "%", "liters", "%", "days", "°C",
        "Unitless (pH)", "Unknown", "% ABV",
    ],
})
table["y"] = range(len(table), 0, -1)
table["fill"] = ["#dbe6ef"] + [
    "#edf4f8" if i < 3 else ("#f4f5f7" if i % 2 == 0 else "#ffffff")
    for i in range(len(table) - 1)
]

plot = (
    ggplot(table, aes(y="y"))
    + geom_rect(aes(xmin=0, xmax=10, ymin="y - 0.46", ymax="y + 0.46", fill="fill"))
    + geom_text(aes(x=0.3, label="variable"), ha="left", size=10, color="#203247")
    + geom_text(aes(x=7.0, label="unit"), ha="left", size=10, color="#203247")
    + scale_fill_identity()
    + xlim(0, 10)
    + ylim(0.4, len(table) + 0.6)
    + labs(
        title="Key brewing variables and units",
        subtitle="Selected for the production-loss research question",
        caption="Loss percentages: denominators undocumented.\n"
        "Source: Kaggle dataset documentation; project data dictionary.",
    )
    + theme_void()
    + theme(
        plot_title=element_text(size=15, weight="bold", color="#203247"),
        plot_subtitle=element_text(size=10, color="#536273"),
        plot_caption=element_text(size=8, ha="left", color="#536273"),
        figure_size=(6.8, 5.4),
    )
)
plot.save(project_dir / "explore" / "variable_units_table.png", dpi=180, verbose=False)


# %% Average production loss
data = pd.read_csv(project_dir / "brewery_data_cleaned.csv")
stages = {
    "Loss_During_Brewing": "Brewing",
    "Loss_During_Fermentation": "Fermentation",
    "Loss_During_Bottling_Kegging": "Bottling/kegging",
}
averages = (
    data[list(stages)]
    .mean()
    .rename(index=stages)
    .rename_axis("Stage")
    .reset_index(name="Average_Loss")
    .assign(
        Stage=lambda frame: pd.Categorical(
            frame["Stage"], categories=list(stages.values()), ordered=True
        ),
        Label=lambda frame: frame["Average_Loss"].map("{:.2f}".format),
    )
)

plot = (
    ggplot(averages, aes(x="Stage", y="Average_Loss", fill="Stage"))
    + geom_col(width=0.6, show_legend=False)
    + geom_text(aes(label="Label"), va="bottom", nudge_y=0.08, size=12)
    + scale_fill_manual(values=["#6489a6", "#244c69", "#88a8bc"])
    + scale_y_continuous(limits=(0, 4.5), breaks=[0, 1, 2, 3, 4])
    + labs(
        title="Average production loss by stage",
        subtitle=f"Unweighted batch averages | {len(data):,} batches",
        x="Production stage",
        y="Average production loss (%)",
        caption="Loss-percentage denominators are undocumented; stage averages are not shares of total loss.",
    )
    + theme_minimal()
    + theme(
        figure_size=(8, 5),
        plot_title=element_text(size=16, weight="bold"),
        axis_text=element_text(size=11),
        plot_caption=element_text(size=8, ha="left"),
    )
)
plot.save(project_dir / "explore" / "average_production_loss.png", dpi=180, verbose=False)
print(averages[["Stage", "Average_Loss"]].to_string(index=False))


# %% Production loss distributions
data = pd.read_csv(project_dir / "brewery_data_cleaned.csv")
stages = {
    "Loss_During_Brewing": "Brewing",
    "Loss_During_Fermentation": "Fermentation",
    "Loss_During_Bottling_Kegging": "Bottling/Kegging",
}
losses = data[list(stages)].rename(columns=stages)
long_data = losses.melt(var_name="Stage", value_name="Loss").assign(
    Stage=lambda frame: pd.Categorical(
        frame["Stage"], categories=list(stages.values()), ordered=True
    )
)

q1 = losses.quantile(0.25)
q3 = losses.quantile(0.75)
iqr = q3 - q1
outliers = losses.lt(q1 - 1.5 * iqr) | losses.gt(q3 + 1.5 * iqr)
summary = pd.DataFrame({
    "Batches": losses.count(),
    "Median": losses.median(),
    "Q1": q1,
    "Q3": q3,
    "IQR": iqr,
    "SD": losses.std(),
    "Minimum": losses.min(),
    "Maximum": losses.max(),
    "Potential_outliers": outliers.sum(),
})

plot = (
    ggplot(long_data, aes(x="Stage", y="Loss", fill="Stage"))
    + geom_boxplot(
        width=0.52, color="#24374a", size=0.75,
        outlier_shape="o", outlier_size=2.8, outlier_alpha=0.8,
        show_legend=False,
    )
    + scale_fill_manual(values=["#91b6cc", "#e4af65", "#83b9aa"])
    + scale_y_continuous(limits=(0, 8), breaks=list(range(9)))
    + labs(
        title="Production loss varies by stage",
        subtitle=f"Distribution across {len(data):,} batches per stage | Fermentation has the highest median",
        x=None,
        y="Production loss (%)",
        caption="Box: middle 50% of batches | Line: median | Whiskers: most extreme values within 1.5 × IQR\n"
        "Points: potential outliers (overlapping points may coincide); all observations retained.\n"
        "Loss denominators are undocumented; stage percentages are not shares of total loss.",
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 6.5),
        plot_title=element_text(size=20, weight="bold", color="#203247"),
        plot_subtitle=element_text(size=11, color="#536273", margin={"b": 18}),
        axis_title_y=element_text(size=12, margin={"r": 12}),
        axis_text=element_text(size=12, color="#203247"),
        panel_grid_major_x=element_blank(),
        panel_grid_minor=element_blank(),
        plot_caption=element_text(size=9, ha="left", color="#536273", margin={"t": 14}),
        plot_background=element_blank(),
    )
)
plot.save(project_dir / "explore" / "production_loss_boxplot.png", dpi=300, verbose=False)

fermentation = losses["Fermentation"]
report = [
    "PRODUCTION LOSS DISTRIBUTIONS",
    "Loss summaries use percent units; IQR and SD use percentage points.",
    summary.round(3).to_string(),
    "\nWITHIN-BATCH COMPARISONS",
]
for stage in ["Brewing", "Bottling/Kegging"]:
    higher = fermentation.gt(losses[stage]).sum()
    tied = fermentation.eq(losses[stage]).sum()
    report.append(
        f"Fermentation > {stage}: {higher}/{len(losses)} ({higher / len(losses):.1%}); "
        f"equal: {tied}/{len(losses)}."
    )
higher_than_both = fermentation.gt(losses[["Brewing", "Bottling/Kegging"]].max(axis=1)).sum()
report.extend([
    f"Fermentation strictly exceeds both other stages: {higher_than_both}/{len(losses)} "
    f"({higher_than_both / len(losses):.1%}).",
    "\nFermentation has the highest median and largest IQR and sample SD. "
    "The distributions overlap; fermentation is not the highest-loss stage in every batch.",
    "Potential outliers are observations strictly beyond the stage-specific 1.5-IQR fences, "
    "not confirmed data errors. No observations were removed.",
    "Comparisons describe recorded percentages in this dataset, not causal effects "
    "or absolute quantities lost. Percentage denominators remain undocumented.",
])
report_text = "\n".join(report) + "\n"
(project_dir / "explore" / "production_loss_boxplot_summary.txt").write_text(report_text, encoding="utf-8")
print(report_text)
