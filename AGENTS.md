# AGENTS.md

This project is used for applied data science work.

## General Working Rules

- When prompted to create script use simple, readable Python.
- Build on existing working code rather than replacing it unnecessarily.
- Avoid unnecessary functions, classes, abstractions, or production-style programming structures.
- Do not perform analyses, diagnostics, transformations, or checks that were not requested.

## Specific Instructions

- Analysis proceeds in stages - explore, preprocess, feature engineering and modeling

- Create folders representing the above stages if not present already. Add empty python script if not present. For example

- `explore/explore.py`
- `preprocess/preprocess.py`
- `features/features.py`
- `model/model.py`

- As work progresses add codes to the relevant script belonging to the relevant analytical stage.

- To handle any data preprocessing, use Pandas. Install it if not installed. Rely on methods chaining rather than creating too many intermediate datasets

- To handle data viz, use plotnine. Install it if not installed. Use a non-interactive Matplotlib backend (`Agg`) for all visualizations. When creating a visualization, save it as an image file.

- To handle regression modeling use formula based interface from statsmodel.

- Never interpret, explain, contextualize, or describe the substantive meaning of model results or coefficients. Only report the requested numerical/model output. Do not translate coefficients into statements such as "an increase in X is associated with an increase/decrease in Y" unless explicitly asked to interpret the results.

- Reporting a fitted equation or coefficient estimate is allowed; explaining what the coefficient means in terms of the variables is interpretation and is prohibited unless explicitly requested.

## Data instructions
- Never modify the raw data.
- Place any new data in the project folder but not in any subfolders

When creating a skill for this project, create only .agents/skills/<skill-name>/SKILL.md; do not create agents/openai.yaml or any other supporting files.