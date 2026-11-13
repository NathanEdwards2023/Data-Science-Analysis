from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
raw_path = project_dir / "brewery_data.csv"
protected_paths = [raw_path, *sorted((project_dir / "explore").rglob("*"))]
original_hashes = {
    path: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in protected_paths if path.is_file()
}

# Read text literally so blanks and missing-value tokens can be audited.
raw = pd.read_csv(raw_path, dtype="string", keep_default_na=False)
text_columns = ["Batch_ID", "Beer_Style", "SKU", "Location", "Ingredient_Ratio"]
category_columns = ["Beer_Style", "SKU", "Location"]
numeric_columns = raw.columns.drop(text_columns + ["Brew_Date"]).tolist()
missing_tokens = ["", "na", "n/a", "nan", "null", "none"]
missing_counts = raw.apply(
    lambda column: column.str.strip().str.lower().isin(missing_tokens).sum()
)
if missing_counts.any():
    raise ValueError("Missing values need review before exporting: " + str(missing_counts))

# Round-trip parsing preserves the original numeric precision.
clean = (
    pd.read_csv(raw_path, dtype={column: "string" for column in text_columns},
                float_precision="round_trip")
    .assign(Brew_Date=lambda frame: pd.to_datetime(
        frame["Brew_Date"], format="%Y-%m-%d", errors="raise"
    ))
)
clean[numeric_columns] = clean[numeric_columns].apply(pd.to_numeric, errors="raise")
if not np.isfinite(clean[numeric_columns].to_numpy()).all():
    raise ValueError("Nonfinite numeric values need review before exporting.")

numeric = clean[numeric_columns]
q1 = numeric.quantile(0.25)
q3 = numeric.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outlier_flags = numeric.lt(lower) | numeric.gt(upper)
range_summary = numeric.describe().T.assign(
    iqr_lower=lower, iqr_upper=upper,
    iqr_flag_count=outlier_flags.sum(), negative_count=numeric.lt(0).sum(),
    zero_count=numeric.eq(0).sum(),
)

report = [
    "PREPROCESSING REPORT",
    f"Input: {raw_path.name}; rows: {len(raw)}; original columns: {len(raw.columns)}",
    "\nMISSING VALUES (including blank and common missing-value tokens)",
    missing_counts.to_string(),
    f"\nExact duplicate rows: {raw.duplicated().sum()}",
    f"Duplicate batch IDs: {clean['Batch_ID'].duplicated().sum()}",
    "All dates parsed using YYYY-MM-DD; all numeric values parsed and are finite.",
    f"Date range: {clean['Brew_Date'].min().date()} to {clean['Brew_Date'].max().date()}",
    "\nNUMERICAL RANGES AND POTENTIAL OUTLIERS",
    range_summary.round(6).to_string(),
    "IQR flags use values strictly outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.",
    "These are pooled screening flags, not evidence of invalid records. "
    "Style/location differences may contribute. No values are removed or capped.",
    "Documented units and remaining unknowns are listed in data_dictionary.md. "
    "No additional range rules or transformations were applied in the documentation update.",
    "\nFLAGGED VALUES (batch ID, column, original value)",
]
for column in numeric_columns:
    flagged = clean.loc[outlier_flags[column], ["Batch_ID", column]]
    if not flagged.empty:
        report.extend([column, flagged.to_string(index=False)])

report.append("\nCATEGORICAL VALIDATION")
for column in category_columns:
    labels = clean[column].drop_duplicates()
    normalized = labels.str.strip().str.casefold()
    report.extend([
        f"\n{column}: {len(labels)} distinct labels",
        clean[column].value_counts().to_string(),
        f"Rows with surrounding whitespace: {clean[column].ne(clean[column].str.strip()).sum()}",
        f"Label collisions after whitespace/case normalization: {normalized.duplicated().sum()}",
    ])

ratio_valid = clean["Ingredient_Ratio"].str.fullmatch(
    r"\d+(?:\.\d+)?:\d+(?:\.\d+)?:\d+(?:\.\d+)?"
)
report.extend([
    "Categories checked for blanks, whitespace and case variants; no external "
    "exhaustive category dictionary is available. Kaggle lists SKU formats but "
    "only examples of beer styles; Location represents a sales region.",
    f"Ingredient_Ratio entries matching three nonnegative numeric components: {ratio_valid.sum()}/{len(clean)}",
    "Ingredient_Ratio retained as text: component identities are undocumented.",
])

report.extend([
    "\nPREPROCESSING DECISIONS",
    "All original rows, row order, column names, category labels and numeric values retained.",
    "No imputation, deduplication, outlier deletion, winsorization, scaling, "
    "categorical encoding, or regression performed.",
    "Batch_ID treated as a string identifier, not a continuous modeling variable.",
    "Brew_Date parsed as datetime and exported in its original YYYY-MM-DD format.",
    "Beer_Style, SKU and Location retained as nominal text labels; no ordinal codes assigned.",
    "No percentile-rank columns are generated; removed at user request.",
    "Original loss variables are percentages; their denominators remain unknown. No total loss, "
    "cross-stage difference/ratio, dominant stage, or volume-weighted loss is derived.",
    "CSV does not preserve dtype metadata. Reload Batch_ID and nominal fields as "
    "strings and parse Brew_Date explicitly. Decide model-specific encoding later.",
    "\nCLEANED IN-MEMORY DATA TYPES",
    clean.dtypes.to_string(),
])

output_path = project_dir / "brewery_data_cleaned.csv"
clean.to_csv(output_path, index=False, date_format="%Y-%m-%d")

# Verify export preserves every original value and all protected source files.
reloaded = pd.read_csv(
    output_path, dtype={column: "string" for column in text_columns},
    parse_dates=["Brew_Date"], float_precision="round_trip",
)
pd.testing.assert_frame_equal(clean, reloaded, check_exact=True)
source = pd.read_csv(
    raw_path, dtype={column: "string" for column in text_columns},
    parse_dates=["Brew_Date"], float_precision="round_trip",
)
pd.testing.assert_frame_equal(source, reloaded[source.columns], check_exact=True)
for path, original_hash in original_hashes.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash, path

report.extend([
    "\nEXPORT VERIFICATION",
    f"Output: {output_path.name}; rows: {len(clean)}; columns: {len(clean.columns)}",
    "CSV round-trip passed; all original values preserved exactly after typed parsing.",
    "SHA-256 checks confirm raw CSV and existing exploration files are unchanged.",
])
report.extend(["\nVARIABLE DEFINITIONS AND UNITS",
               (project_dir / "data_dictionary.md").read_text(encoding="utf-8")])
report_text = "\n".join(report) + "\n"
(project_dir / "preprocess" / "preprocessing_report.txt").write_text(
    report_text, encoding="utf-8"
)
print(report_text)
