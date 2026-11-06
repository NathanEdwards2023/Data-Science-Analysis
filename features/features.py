"""Create an APA-style summary chart for Total Sales."""

import csv
import os
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "brewery_data_complete_extended.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "total_sales_mean_sd_apa.png")
TABLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "total_sales_descriptive_table.csv")
TABLE_PNG_PATH = os.path.join(os.path.dirname(__file__), "total_sales_descriptive_table.png")
DESCRIPTIVE_CSV_PATH = os.path.join(os.path.dirname(__file__), "descriptive_statistics_selected_variables.csv")
DESCRIPTIVE_PNG_PATH = os.path.join(os.path.dirname(__file__), "descriptive_statistics_selected_variables.png")
SALES_ALCOHOL_CSV_PATH = os.path.join(os.path.dirname(__file__), "descriptive_statistics_sales_alcohol.csv")
SALES_ALCOHOL_PNG_PATH = os.path.join(os.path.dirname(__file__), "descriptive_statistics_sales_alcohol_apa.png")
SALES_BY_ALCOHOL_CSV_PATH = os.path.join(os.path.dirname(__file__), "total_sales_by_alcohol_content.csv")
SALES_BY_ALCOHOL_PNG_PATH = os.path.join(os.path.dirname(__file__), "total_sales_by_alcohol_content_apa.png")
BAR_CHART_PATH = os.path.join(os.path.dirname(__file__), "beer_style_average_total_sales.png")
BEER_STYLE_HORIZONTAL_BAR_PATH = os.path.join(os.path.dirname(__file__), "total_sales_by_beer_style_horizontal_bar.png")
PACKAGE_TYPE_BAR_PATH = os.path.join(os.path.dirname(__file__), "total_sales_by_package_type_bar.png")
CATEGORICAL_SALES_PERCENT_PATH = os.path.join(os.path.dirname(__file__), "categorical_total_sales_percent.csv")
CATEGORICAL_SALES_PERCENT_PNG_PATH = os.path.join(os.path.dirname(__file__), "categorical_total_sales_percent_apa.png")
CAT_VARIABLES = [("Beer_Style", "Beer Style"), ("SKU", "Package Type"), ("Location", "Location")]
COMBINED_CATEGORICAL_PIE_PATH = os.path.join(os.path.dirname(__file__), "categorical_sales_percent_pies_combined.png")
CONTINUOUS_DESCRIPTIVE_PATH = os.path.join(os.path.dirname(__file__), "continuous_descriptive_mean_sd.csv")
CONTINUOUS_DESCRIPTIVE_PNG_PATH = os.path.join(os.path.dirname(__file__), "continuous_descriptive_mean_sd_apa.png")
BOX_PLOT_PATH = os.path.join(os.path.dirname(__file__), "total_sales_by_packaging_type_boxplot.png")
SCATTER_PLOT_PATH = os.path.join(os.path.dirname(__file__), "total_sales_vs_volume_produced.png")
ALCOHOL_SCATTER_PLOT_PATH = os.path.join(os.path.dirname(__file__), "total_sales_vs_alcohol_content.png")
ALCOHOL_LINE_PLOT_PATH = os.path.join(os.path.dirname(__file__), "total_sales_line_by_alcohol_content.png")
VOLUME_LINE_PLOT_PATH = os.path.join(os.path.dirname(__file__), "total_sales_line_by_volume_produced.png")
SALES_ALCOHOL_VOLUME_SCATTER_PATH = os.path.join(
    os.path.dirname(__file__), "total_sales_by_alcohol_content_and_volume_produced.png"
)
HEATMAP_PATH = os.path.join(os.path.dirname(__file__), "beer_style_location_average_total_sales_heatmap.png")
ALL_CATEGORICAL_HEATMAP_PATH = os.path.join(os.path.dirname(__file__), "all_categorical_variables_total_sales_heatmap.png")
SAMPLE_SIZE = 100_000
SEED = 767


def reservoir_sample(path, sample_size, seed):
    """Read the large CSV once and retain a reproducible random sample."""
    rng = random.Random(seed)
    sample = []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader):
            if row_number < sample_size:
                sample.append(row)
            else:
                replacement = rng.randint(0, row_number)
                if replacement < sample_size:
                    sample[replacement] = row
    return pd.DataFrame(sample)


def main():
    # Load only the variables needed for the requested summary chart.
    data = reservoir_sample(DATA_PATH, SAMPLE_SIZE, SEED)
    data["Total_Sales"] = data["Total_Sales"].astype(float)
    data["Volume_Produced"] = data["Volume_Produced"].astype(float)
    data["Alcohol_Content"] = data["Alcohol_Content"].astype(float)

    # Create the requested visualizations from the sampled data.
    beer_style_means = data.groupby("Beer_Style")["Total_Sales"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(beer_style_means.index, beer_style_means.values, color="0.35")
    ax.set_xlabel("Beer Style")
    ax.set_ylabel("Average Total Sales")
    ax.set_title("Average Total Sales by Beer Style")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="0.85")
    fig.tight_layout()
    fig.savefig(BAR_CHART_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Compare mean Total Sales across beer styles with a horizontal bar chart.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(beer_style_means.index, beer_style_means.values, color="0.35")
    ax.set_xlabel("Average Total Sales")
    ax.set_ylabel("Beer Style")
    ax.set_title("Average Total Sales by Beer Style")
    ax.grid(axis="x", color="0.85")
    fig.tight_layout()
    fig.savefig(BEER_STYLE_HORIZONTAL_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Compare mean Total Sales across package types with a bar chart.
    package_type_means = data.groupby("SKU")["Total_Sales"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(package_type_means.index, package_type_means.values, color="0.35")
    ax.set_xlabel("Package Type")
    ax.set_ylabel("Average Total Sales")
    ax.set_title("Average Total Sales by Package Type")
    ax.grid(axis="y", color="0.85")
    fig.tight_layout()
    fig.savefig(PACKAGE_TYPE_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Summarize total sales and percentage of overall sales for categorical variables.
    categorical_tables = []
    for variable, label in [("Beer_Style", "Beer Style"), ("SKU", "Package Type"), ("Location", "Location")]:
        summary = data.groupby(variable)["Total_Sales"].sum().reset_index()
        summary.columns = ["Category", "Total Sales"]
        summary["Variable"] = label
        summary["Sales Percentage"] = summary["Total Sales"] / summary["Total Sales"].sum() * 100
        categorical_tables.append(summary[["Variable", "Category", "Sales Percentage"]])
    categorical_sales_percent = pd.concat(categorical_tables, ignore_index=True)
    categorical_sales_percent["Sales Percentage"] = categorical_sales_percent["Sales Percentage"].round(2)
    categorical_sales_percent.to_csv(CATEGORICAL_SALES_PERCENT_PATH, index=False)

    # Create separate APA tables and colorful pie charts for each categorical variable.
    for variable, label in CAT_VARIABLES:
        variable_table = categorical_sales_percent[categorical_sales_percent["Variable"] == label].copy()
        safe_label = label.lower().replace(" ", "_")
        table_path = os.path.join(os.path.dirname(__file__), f"{safe_label}_total_sales_percent_apa.png")
        pie_path = os.path.join(os.path.dirname(__file__), f"{safe_label}_sales_percent_pie.png")

        fig_table, ax_table = plt.subplots(figsize=(6, 3.5))
        ax_table.axis("off")
        display_variable = variable_table.copy()
        display_variable["Sales Percentage"] = display_variable["Sales Percentage"].map(lambda value: f"{value:.2f}%")
        table = ax_table.table(
            cellText=display_variable[["Category", "Sales Percentage"]].values,
            colLabels=[label, "Sales Percentage"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.0 if row == 0 else 0.25)
            if row == 0:
                cell.set_text_props(weight="bold")
        fig_table.text(0.01, 0.94, f"Table. Sales Percentage by {label}", ha="left", fontsize=12, weight="bold")
        fig_table.text(0.01, 0.03, "Note. Percentages represent each category's share of sales within the variable.", ha="left", fontsize=8, style="italic")
        fig_table.tight_layout(rect=[0, 0.08, 1, 0.9])
        fig_table.savefig(table_path, dpi=300, bbox_inches="tight")
        plt.close(fig_table)

        fig_pie, ax_pie = plt.subplots(figsize=(7, 6))
        ax_pie.pie(
            variable_table["Sales Percentage"],
            labels=variable_table["Category"],
            autopct="%.1f%%",
            startangle=90,
            colors=plt.cm.tab20.colors[:len(variable_table)],
            wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        )
        ax_pie.set_title(f"Sales Percentage by {label}")
        fig_pie.tight_layout()
        fig_pie.savefig(pie_path, dpi=300, bbox_inches="tight")
        plt.close(fig_pie)

    # Combine all selected categorical variables into one comparison figure.
    fig_combined, axes_combined = plt.subplots(1, 3, figsize=(18, 6))
    for ax_pie, (variable, label) in zip(axes_combined, CAT_VARIABLES):
        variable_table = categorical_sales_percent[categorical_sales_percent["Variable"] == label]
        ax_pie.pie(
            variable_table["Sales Percentage"],
            labels=variable_table["Category"],
            autopct="%.1f%%",
            startangle=90,
            colors=plt.cm.tab20.colors[:len(variable_table)],
            wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        )
        ax_pie.set_title(label)
    fig_combined.suptitle("Percentage of Total Sales by Categorical Variables")
    fig_combined.tight_layout()
    fig_combined.savefig(COMBINED_CATEGORICAL_PIE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_combined)

    # Create an APA-style table image for the categorical sales summary.
    fig_table, ax_table = plt.subplots(figsize=(10, 9))
    ax_table.axis("off")
    display_table = categorical_sales_percent.copy()
    display_table["Sales Percentage"] = display_table["Sales Percentage"].map(lambda value: f"{value:.2f}%")
    table = ax_table.table(
        cellText=display_table.values,
        colLabels=["Variable", "Category", "Sales Percentage"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0 if row == 0 else 0.25)
        if row == 0:
            cell.set_text_props(weight="bold")
    fig_table.text(0.01, 0.96, "Table. Total Sales by Categorical Variable", ha="left", fontsize=12, weight="bold")
    fig_table.text(0.01, 0.02, "Note. Sales Percentage represents each category's share of total sales within its categorical variable.", ha="left", fontsize=8, style="italic")
    fig_table.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig_table.savefig(CATEGORICAL_SALES_PERCENT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_table)

    # Create APA-style descriptive statistics for the continuous analysis variables.
    data["Alcohol Content Group"] = pd.qcut(data["Alcohol_Content"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    data["Volume Produced Group"] = pd.qcut(data["Volume_Produced"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    continuous_tables = []
    for group_column, label in [("Alcohol Content Group", "Alcohol Content"), ("Volume Produced Group", "Volume Produced")]:
        grouped = data.groupby(group_column, observed=False)["Total_Sales"].agg(N="count", Mean="mean", SD="std").reset_index()
        grouped.columns = ["Group", "N", "Mean", "SD"]
        grouped.insert(0, "Variable", label)
        grouped["Mean (SD)"] = grouped.apply(lambda row: f"{row['Mean']:,.2f} ({row['SD']:,.2f})", axis=1)
        continuous_tables.append(grouped[["Variable", "Group", "N", "Mean (SD)"]])
    continuous = pd.concat(continuous_tables, ignore_index=True)
    continuous.to_csv(CONTINUOUS_DESCRIPTIVE_PATH, index=False)

    fig_table, ax_table = plt.subplots(figsize=(8, 4.5))
    ax_table.axis("off")
    table = ax_table.table(
        cellText=continuous[["Variable", "Group", "N", "Mean (SD)"]].values,
        colLabels=["Variable", "Group", "N", "Mean (SD)"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0 if row == 0 else 0.25)
        if row == 0:
            cell.set_text_props(weight="bold")
    fig_table.text(0.01, 0.94, "Table. Descriptive Summary of Total Sales, Alcohol Content, and Volume Produced", ha="left", fontsize=12, weight="bold")
    fig_table.text(0.01, 0.03, "Note. Values are reported as Mean (SD).", ha="left", fontsize=8, style="italic")
    fig_table.tight_layout(rect=[0, 0.08, 1, 0.9])
    fig_table.savefig(CONTINUOUS_DESCRIPTIVE_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_table)

    packaging_groups = [group["Total_Sales"].values for _, group in data.groupby("SKU")]
    packaging_labels = list(data.groupby("SKU").groups.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(packaging_groups, tick_labels=packaging_labels, patch_artist=True, boxprops={"facecolor": "0.75"}, medianprops={"color": "black"})
    ax.set_xlabel("Packaging Type")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales by Packaging Type")
    ax.grid(axis="y", color="0.85")
    fig.tight_layout()
    fig.savefig(BOX_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["Volume_Produced"], data["Total_Sales"], s=4, alpha=0.2, color="0.2")
    ax.set_xlabel("Volume Produced")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales and Volume Produced")
    ax.grid(color="0.85")
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["Alcohol_Content"], data["Total_Sales"], s=4, alpha=0.2, color="0.2")
    ax.set_xlabel("Alcohol Content")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales by Alcohol Content")
    ax.grid(color="0.85")
    fig.tight_layout()
    fig.savefig(ALCOHOL_SCATTER_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Show mean Total Sales across ordered bins for readable line graphs.
    data["Alcohol Content Bin"] = pd.qcut(data["Alcohol_Content"], q=20, duplicates="drop")
    alcohol_line = data.groupby("Alcohol Content Bin", observed=False).agg(
        Predictor=("Alcohol_Content", "mean"), Mean_Total_Sales=("Total_Sales", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alcohol_line["Predictor"], alcohol_line["Mean_Total_Sales"], marker="o", color="0.2")
    ax.set_xlabel("Mean Alcohol Content within Bin")
    ax.set_ylabel("Mean Total Sales")
    ax.set_title("Mean Total Sales by Alcohol Content")
    ax.grid(color="0.85")
    fig.tight_layout()
    fig.savefig(ALCOHOL_LINE_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    data["Volume Produced Bin"] = pd.qcut(data["Volume_Produced"], q=20, duplicates="drop")
    volume_line = data.groupby("Volume Produced Bin", observed=False).agg(
        Predictor=("Volume_Produced", "mean"), Mean_Total_Sales=("Total_Sales", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(volume_line["Predictor"], volume_line["Mean_Total_Sales"], marker="o", color="0.2")
    ax.set_xlabel("Mean Volume Produced within Bin")
    ax.set_ylabel("Mean Total Sales")
    ax.set_title("Mean Total Sales by Volume Produced")
    ax.grid(color="0.85")
    fig.tight_layout()
    fig.savefig(VOLUME_LINE_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Compare Total Sales with the two requested continuous predictors.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(data["Alcohol_Content"], data["Total_Sales"], s=4, alpha=0.2, color="0.2")
    axes[0].set_xlabel("Alcohol Content")
    axes[0].set_ylabel("Total Sales")
    axes[0].set_title("Total Sales by Alcohol Content")
    axes[0].grid(color="0.85")

    axes[1].scatter(data["Volume_Produced"], data["Total_Sales"], s=4, alpha=0.2, color="0.2")
    axes[1].set_xlabel("Volume Produced")
    axes[1].set_ylabel("Total Sales")
    axes[1].set_title("Total Sales by Volume Produced")
    axes[1].grid(color="0.85")

    fig.suptitle("Total Sales by Alcohol Content and Volume Produced")
    fig.tight_layout()
    fig.savefig(SALES_ALCOHOL_VOLUME_SCATTER_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    heatmap = data.pivot_table(index="Beer_Style", columns="Location", values="Total_Sales", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.set_xlabel("Location")
    ax.set_ylabel("Beer Style")
    ax.set_title("Average Total Sales by Beer Style and Location")
    fig.colorbar(image, ax=ax, label="Average Total Sales")
    fig.tight_layout()
    fig.savefig(HEATMAP_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Combine all categorical variables into one heat map of average Total Sales.
    categorical_heatmap_parts = []
    for variable, label in CAT_VARIABLES:
        part = data.groupby(variable)["Total_Sales"].mean().reset_index()
        part.columns = ["Category", "Average Total Sales"]
        part["Variable"] = label
        part["Row Label"] = part["Variable"] + ": " + part["Category"]
        categorical_heatmap_parts.append(part)
    all_categorical = pd.concat(categorical_heatmap_parts, ignore_index=True)
    all_categorical_heatmap = all_categorical.set_index("Row Label")[["Average Total Sales"]]
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(all_categorical_heatmap.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks([0], ["Average Total Sales"])
    ax.set_yticks(range(len(all_categorical_heatmap.index)), all_categorical_heatmap.index)
    ax.set_ylabel("Categorical Variable and Category")
    ax.set_title("Average Total Sales Across All Categorical Variables")
    for row_index, value in enumerate(all_categorical_heatmap["Average Total Sales"]):
        ax.text(0, row_index, f"{value:,.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Average Total Sales")
    fig.tight_layout()
    fig.savefig(ALL_CATEGORICAL_HEATMAP_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Summarize Total Sales by Alcohol Content quartile because Alcohol Content is continuous.
    data["Alcohol Content Group"] = pd.qcut(
        data["Alcohol_Content"], q=4,
        labels=["Q1 (lowest alcohol)", "Q2", "Q3", "Q4 (highest alcohol)"],
    )
    sales_by_alcohol = (
        data.groupby("Alcohol Content Group", observed=False)["Total_Sales"]
        .agg(n="count", Mean="mean", SD="std")
        .reset_index()
    )
    sales_by_alcohol["Mean"] = sales_by_alcohol["Mean"].round(2)
    sales_by_alcohol["SD"] = sales_by_alcohol["SD"].round(2)
    sales_by_alcohol.to_csv(SALES_BY_ALCOHOL_CSV_PATH, index=False)

    fig_by_alcohol, ax_by_alcohol = plt.subplots(figsize=(8, 2.8))
    ax_by_alcohol.axis("off")
    display_by_alcohol = sales_by_alcohol.copy()
    display_by_alcohol["n"] = display_by_alcohol["n"].map(lambda value: f"{int(value):,}")
    display_by_alcohol["Mean"] = display_by_alcohol["Mean"].map(lambda value: f"{value:,.2f}")
    display_by_alcohol["SD"] = display_by_alcohol["SD"].map(lambda value: f"{value:,.2f}")
    apa_by_alcohol = ax_by_alcohol.table(
        cellText=display_by_alcohol.values,
        colLabels=["Alcohol Content Group", "n", "M", "SD"],
        cellLoc="left", colLoc="left", loc="center",
    )
    apa_by_alcohol.auto_set_font_size(False)
    apa_by_alcohol.set_fontsize(9)
    apa_by_alcohol.scale(1, 1.5)
    for (row, column), cell in apa_by_alcohol.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0 if row == 0 else 0.25)
        if row == 0:
            cell.set_text_props(weight="bold")
    fig_by_alcohol.text(0.01, 0.94, "Table. Total Sales by Alcohol Content", ha="left", fontsize=11, weight="bold")
    fig_by_alcohol.text(0.01, 0.02, "Note. M = mean; SD = standard deviation. Alcohol Content is grouped into quartiles.", ha="left", fontsize=8, style="italic")
    fig_by_alcohol.tight_layout(rect=[0, 0.1, 1, 0.9])
    fig_by_alcohol.savefig(SALES_BY_ALCOHOL_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_by_alcohol)

    # Create descriptive statistics for Total Sales and Alcohol Content.
    sales_alcohol = pd.DataFrame([
        ["Total Sales", data["Total_Sales"].count(), data["Total_Sales"].mean(), data["Total_Sales"].std(), data["Total_Sales"].min(), data["Total_Sales"].median(), data["Total_Sales"].max()],
        ["Alcohol Content", data["Alcohol_Content"].count(), data["Alcohol_Content"].mean(), data["Alcohol_Content"].std(), data["Alcohol_Content"].min(), data["Alcohol_Content"].median(), data["Alcohol_Content"].max()],
    ], columns=["Variable", "N", "Mean", "SD", "Minimum", "Median", "Maximum"])
    sales_alcohol.iloc[:, 2:] = sales_alcohol.iloc[:, 2:].round(2)
    sales_alcohol.to_csv(SALES_ALCOHOL_CSV_PATH, index=False)

    fig_sales_alcohol, ax_sales_alcohol = plt.subplots(figsize=(9, 2.4))
    ax_sales_alcohol.axis("off")
    display_sales_alcohol = sales_alcohol.copy()
    for column in ["Mean", "SD", "Minimum", "Median", "Maximum"]:
        display_sales_alcohol[column] = display_sales_alcohol[column].map(lambda value: f"{value:,.2f}")
    display_sales_alcohol["N"] = display_sales_alcohol["N"].map(lambda value: f"{int(value):,}")
    apa_sales_alcohol = ax_sales_alcohol.table(
        cellText=display_sales_alcohol.values,
        colLabels=["Variable", "N", "M", "SD", "Min", "Median", "Max"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    apa_sales_alcohol.auto_set_font_size(False)
    apa_sales_alcohol.set_fontsize(9)
    apa_sales_alcohol.scale(1, 1.6)
    for (row, column), cell in apa_sales_alcohol.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0 if row == 0 else 0.25)
        if row == 0:
            cell.set_text_props(weight="bold")
    fig_sales_alcohol.text(0.01, 0.94, "Table. Descriptive Statistics for Total Sales and Alcohol Content", ha="left", fontsize=11, weight="bold")
    fig_sales_alcohol.text(0.01, 0.02, "Note. M = mean; SD = standard deviation. Results use a reproducible sample of 100,000 observations.", ha="left", fontsize=8, style="italic")
    fig_sales_alcohol.tight_layout(rect=[0, 0.1, 1, 0.9])
    fig_sales_alcohol.savefig(SALES_ALCOHOL_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_sales_alcohol)

    # Build descriptive statistics for the outcome and all selected predictors.
    descriptive_rows = []
    for variable in ["Total_Sales", "Volume_Produced"]:
        values = data[variable]
        descriptive_rows.append({
            "Variable": variable,
            "Type": "Numeric",
            "N": values.notna().sum(),
            "Mean": values.mean(),
            "SD": values.std(),
            "Minimum": values.min(),
            "Median": values.median(),
            "Maximum": values.max(),
            "Categories": "",
        })
    for variable in ["Beer_Style", "SKU", "Location"]:
        values = data[variable]
        descriptive_rows.append({
            "Variable": variable,
            "Type": "Categorical",
            "N": values.notna().sum(),
            "Mean": "",
            "SD": "",
            "Minimum": "",
            "Median": "",
            "Maximum": "",
            "Categories": values.nunique(),
        })
    descriptive_table = pd.DataFrame(descriptive_rows)
    descriptive_table.loc[:, ["Mean", "SD", "Minimum", "Median", "Maximum"]] = descriptive_table[
        ["Mean", "SD", "Minimum", "Median", "Maximum"]
    ].apply(pd.to_numeric, errors="coerce").round(2)
    descriptive_table.to_csv(DESCRIPTIVE_CSV_PATH, index=False)

    fig_stats, ax_stats = plt.subplots(figsize=(12, 3.2))
    ax_stats.axis("off")
    stats_display = descriptive_table.fillna("")
    table_stats = ax_stats.table(
        cellText=stats_display.values,
        colLabels=["Variable", "Type", "N", "M", "SD", "Min", "Median", "Max", "Categories"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table_stats.auto_set_font_size(False)
    table_stats.set_fontsize(8)
    table_stats.scale(1, 1.6)
    for (row, column), cell in table_stats.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0 if row == 0 else 0.25)
        if row == 0:
            cell.set_text_props(weight="bold")
    fig_stats.text(0.01, 0.02, "Note. M = mean; SD = standard deviation. Numeric summaries use a reproducible sample of 100,000 observations.", ha="left", fontsize=8)
    fig_stats.tight_layout(rect=[0, 0.08, 1, 1])
    fig_stats.savefig(DESCRIPTIVE_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_stats)

    # Quartiles convert continuous production volume into four readable groups.
    data["Production Volume"] = pd.qcut(
        data["Volume_Produced"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )

    grouping_variables = {
        "Beer Style": "Beer_Style",
        "Packaging Type": "SKU",
        "Location": "Location",
        "Production Volume": "Production Volume",
    }
    summaries = []
    for label, column in grouping_variables.items():
        summary = (
            data.groupby(column, observed=False)["Total_Sales"]
            .agg(Mean="mean", SD="std", n="count")
            .reset_index()
            .rename(columns={column: "Group"})
        )
        summary["Variable"] = label
        summaries.append(summary)

    chart_data = pd.concat(summaries, ignore_index=True)
    chart_data["Panel"] = chart_data["Variable"] + ": " + chart_data["Group"].astype(str)

    # Save the APA-style descriptive table with rounded reporting values.
    table_data = chart_data[["Variable", "Group", "n", "Mean", "SD"]].copy()
    table_data["Mean"] = table_data["Mean"].round(2)
    table_data["SD"] = table_data["SD"].round(2)
    table_data.to_csv(TABLE_CSV_PATH, index=False)

    # Save one separate descriptive table for each independent variable.
    for variable in table_data["Variable"].unique():
        variable_file_name = variable.lower().replace(" ", "_")
        variable_table = table_data[table_data["Variable"] == variable][["Group", "n", "Mean", "SD"]]
        variable_table.to_csv(
            os.path.join(os.path.dirname(__file__), f"total_sales_by_{variable_file_name}.csv"),
            index=False,
        )
        # Create an APA-style image for the individual variable table.
        fig_variable, ax_variable = plt.subplots(figsize=(8, max(2.5, 0.55 * (len(variable_table) + 2))))
        ax_variable.axis("off")
        display_table = variable_table.copy()
        display_table["Mean"] = display_table["Mean"].map(lambda value: f"{value:,.2f}")
        display_table["SD"] = display_table["SD"].map(lambda value: f"{value:,.2f}")
        display_table["n"] = display_table["n"].map(lambda value: f"{value:,}")
        apa_table = ax_variable.table(
            cellText=display_table.values,
            colLabels=["Category", "n", "M", "SD"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        apa_table.auto_set_font_size(False)
        apa_table.set_fontsize(10)
        apa_table.scale(1, 1.45)
        for (row, column), cell in apa_table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.0 if row == 0 else 0.25)
            if row == 0:
                cell.set_text_props(weight="bold")
        fig_variable.text(0.01, 0.96, f"Table. Total Sales by {variable}", ha="left", fontsize=12, weight="bold")
        fig_variable.text(0.01, 0.02, "Note. M = mean; SD = standard deviation. Production Volume is grouped into quartiles.", ha="left", fontsize=8, style="italic")
        fig_variable.tight_layout(rect=[0, 0.08, 1, 0.92])
        fig_variable.savefig(
            os.path.join(os.path.dirname(__file__), f"total_sales_by_{variable_file_name}_apa.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_variable)

    fig_table, ax_table = plt.subplots(figsize=(10, 8))
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_data.values,
        colLabels=["Variable", "Group", "n", "M", "SD"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.0)
        else:
            cell.set_linewidth(0.25)
    fig_table.text(0.01, 0.02, "Note. M = mean; SD = standard deviation. Production Volume is grouped into quartiles.", ha="left", fontsize=8)
    fig_table.tight_layout(rect=[0, 0.05, 1, 1])
    fig_table.savefig(TABLE_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_table)

    # APA-style design: white background, black marks, no chart title, and clear labels.
    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(chart_data))
    ax.errorbar(
        x,
        chart_data["Mean"],
        yerr=chart_data["SD"],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3,
        markersize=4,
        linewidth=1,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(chart_data["Panel"], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Grouping variable and category")
    ax.set_ylabel("Mean Total Sales (error bars = ±1 SD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=0.7)
    fig.text(
        0.01,
        0.01,
        "Note. Production Volume is grouped into sample quartiles. Values are based on a reproducible sample of 100,000 observations.",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved APA-style chart to: {OUTPUT_PATH}")
    print(f"Saved descriptive table to: {TABLE_CSV_PATH}")
    print(f"Saved table image to: {TABLE_PNG_PATH}")
    print(f"Saved selected-variable statistics to: {DESCRIPTIVE_CSV_PATH}")
    print(f"Saved statistics table image to: {DESCRIPTIVE_PNG_PATH}")
    print(f"Saved sales/alcohol statistics to: {SALES_ALCOHOL_CSV_PATH}")
    print(f"Saved APA sales/alcohol table image to: {SALES_ALCOHOL_PNG_PATH}")
    print(f"Saved Total Sales by Alcohol Content to: {SALES_BY_ALCOHOL_CSV_PATH}")
    print(f"Saved APA Total Sales by Alcohol Content table to: {SALES_BY_ALCOHOL_PNG_PATH}")
    print(f"Saved bar chart to: {BAR_CHART_PATH}")
    print(f"Saved horizontal beer style bar chart to: {BEER_STYLE_HORIZONTAL_BAR_PATH}")
    print(f"Saved package type bar chart to: {PACKAGE_TYPE_BAR_PATH}")
    print(f"Saved categorical sales percentage table to: {CATEGORICAL_SALES_PERCENT_PATH}")
    print(f"Saved APA categorical sales table to: {CATEGORICAL_SALES_PERCENT_PNG_PATH}")
    print("Saved separate categorical APA tables and colorful pie charts.")
    print(f"Saved combined categorical pie charts to: {COMBINED_CATEGORICAL_PIE_PATH}")
    print(f"Saved continuous descriptive table to: {CONTINUOUS_DESCRIPTIVE_PATH}")
    print(f"Saved APA continuous descriptive table to: {CONTINUOUS_DESCRIPTIVE_PNG_PATH}")
    print(f"Saved box plot to: {BOX_PLOT_PATH}")
    print(f"Saved scatter plot to: {SCATTER_PLOT_PATH}")
    print(f"Saved alcohol content scatter plot to: {ALCOHOL_SCATTER_PLOT_PATH}")
    print(f"Saved alcohol content line plot to: {ALCOHOL_LINE_PLOT_PATH}")
    print(f"Saved volume produced line plot to: {VOLUME_LINE_PLOT_PATH}")
    print(f"Saved alcohol/volume scatter plot to: {SALES_ALCOHOL_VOLUME_SCATTER_PATH}")
    print(f"Saved heat map to: {HEATMAP_PATH}")
    print(f"Saved all-categorical heat map to: {ALL_CATEGORICAL_HEATMAP_PATH}")
    print(chart_data[["Variable", "Group", "Mean", "SD", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
