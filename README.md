# FIFA World Cup — Linear Regression and Multicollinearity Analysis

**Predicting Total Goals per Match | 1930 – 2018**

---

## Overview

This project applies linear regression to FIFA World Cup match data spanning 88 years and 900 matches. The central objective is not just to build a predictive model, but to deliberately introduce multicollinearity into the feature set, detect it through multiple diagnostic methods, and systematically resolve it using three remediation techniques — feature removal, Ridge regularisation, and Principal Component Analysis.

The result is a structured, end-to-end study of how multicollinearity silently corrupts regression models, and what a responsible data scientist can do about it.

---

## Dataset

| File | Description | Shape |
|------|-------------|-------|
| `world_cup_matches.csv` | Match-level data: teams, scores, stage, host flag | 900 rows × 10 columns |
| `world_cups.csv` | Tournament-level data: winner, goals scored, teams qualified | 22 rows × 9 columns |

**Source:** FIFA World Cup public dataset (Kaggle)

**Target variable:** `Total_Goals` — home goals plus away goals per match

**Features engineered:**

| Feature | Formula | Collinearity Risk |
|---------|---------|------------------|
| `Total_Goals` | Home Goals + Away Goals | Target |
| `Goal_Diff` | Home Goals − Away Goals | Perfect collinearity with Home/Away Goals |
| `Is_Host` | Host Team flag cast to int | Low |
| `Is_KO` | Knockout vs Group Stage | Low |
| `Era` | Binned decade: 1929–2018 into 4 bands | Low |
| `Avg_Goals_WC` | WC_Goals / WC_Matches | Correlated with Goals_Per_Team |
| `Goals_Per_Team` | WC_Goals / WC_Teams | Correlated with Avg_Goals_WC |

---

## Methodology

### Step 1 — Exploratory Data Analysis

- Distribution of `Total_Goals` (histogram + KDE, mean/median marked)
- Average goals per match trend across World Cup years (1930 – 2018)
- Group Stage versus Knockout stage goal distributions (boxplot comparison)
- Home Goals vs Away Goals scatter plot coloured by `Total_Goals`
- Full correlation heatmap for all features — visually exposing the perfect linear relationship between `Goal_Diff`, `Home Goals`, and `Away Goals`

### Step 2 — OLS Regression and R² / Adjusted R² Analysis

An Ordinary Least Squares model is fitted with the full 8-feature collinear set. R² and Adjusted R² are then tracked as features are progressively added, from 1 feature to the full set. The widening gap between R² and Adjusted R² as collinear features enter is a key diagnostic signal — this experiment makes that gap explicit and visible.

Key observation: the full collinear model reports R² = 1.000, which appears perfect. The condition number reveals the truth.

### Step 3 — Multicollinearity Detection via VIF

The Variance Inflation Factor is computed for every feature in the collinear set.

```
VIF Interpretation
------------------
VIF = 1          No multicollinearity
VIF 1 – 5        Acceptable
VIF 5 – 10       Moderate — investigate
VIF > 10         Severe — must address
VIF = infinity   Perfect collinearity — singular matrix
```

Findings from the collinear model:
- `Goal_Diff`, `Home Goals`, `Away Goals` → VIF = infinity (perfect linear dependency)
- `Avg_Goals_WC`, `Goals_Per_Team` → VIF > 10 (both derived from the same tournament totals)
- Condition Number >> 1,000 confirming near-singular design matrix

An eigenvalue analysis of X'X is also included: near-zero eigenvalues confirm the singular structure.

### Step 4 — Handling Multicollinearity (Three Methods)

**Method 1 — Feature Removal**

Two root-cause features are removed:
- `Goal_Diff` (exact linear combination of Home Goals and Away Goals — perfect collinearity by construction)
- `Goals_Per_Team` (derived from same tournament totals as `Avg_Goals_WC`)

Remaining 6-feature set is refitted with OLS. Max VIF drops from infinity to approximately 7.3.

**Method 2 — Ridge Regression (L2 Regularisation)**

Ridge is applied to both the collinear set and the cleaned set across a log-scale grid of alpha values: 0.001, 0.01, 0.1, 1, 10, 100. A 5-fold cross-validation determines the best alpha per set. Ridge does not eliminate the collinearity structurally but shrinks inflated coefficients, restoring stability to inference.

**Method 3 — PCA Regression**

The collinear feature set is standardised and projected into 4 principal components. Because principal components are orthogonal by construction, the VIF of every component equals exactly 1. An OLS model is then fitted on the PCA-transformed inputs, eliminating multicollinearity at the cost of interpretability.

### Step 5 — Before vs After Comparison

A systematic before/after comparison is carried out across three dimensions:

**Coefficient stability:** The same shared features are compared across the collinear and cleaned OLS models. The Standard Error ratio (collinear SE / cleaned SE) quantifies how much multicollinearity inflated each coefficient's uncertainty. A ratio greater than 1 means the collinear model's inference is unreliable for that feature.

**Model comparison table:**

| Model | R² | Adj R² | Max VIF | Stable |
|-------|----|--------|---------|--------|
| OLS — Collinear (8 features) | 1.0000 | 1.0000 | infinity | No — singular |
| OLS — Cleaned (6 features) | ~0.92 | ~0.92 | ~7.3 | Yes |
| PCA Regression (4 components) | ~0.92 | ~0.92 | 1.0 (orthogonal) | Yes |
| Ridge — Collinear set | — | — | infinity (input) | Yes — regularised |
| Ridge — Cleaned set | — | — | ~7.3 | Best overall |

**Residual distributions** for all three valid models are overlaid to confirm comparable error structure after remediation.

---

## Key Findings

**A perfect R² is not a reliable signal of a good model.** When `Goal_Diff` (= Home Goals − Away Goals) is included as a feature alongside `Home Goals` and `Away Goals`, the design matrix X becomes exactly singular. OLS solves this degenerate system and reports R² = 1.000 — which is mathematically true but statistically meaningless. The coefficient estimates are unstable and the standard errors are so inflated that no inference from p-values or confidence intervals can be trusted.

**VIF and condition number are non-negotiable diagnostics.** Neither R² nor residual plots expose this problem. Only VIF (infinite for the collinear features) and the condition number (order 10^15) reveal the severity.

**Removing the root cause is more reliable than regularisation alone.** Ridge stabilises the collinear model but does not fix the underlying structural problem. The most robust outcome — stable coefficients, acceptable VIF, and good cross-validated R² — comes from combining feature removal with Ridge regularisation on the cleaned set.

---

## Repository Structure

```
.
├── WorldCup_Regression.ipynb     # Main analysis notebook (self-contained)
├── world_cup_matches.csv         # Match-level dataset (900 rows)
├── world_cups.csv                # Tournament-level dataset (22 rows)
└── README.md
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

---

## Running the Notebook

1. Clone the repository
2. Place `world_cup_matches.csv` and `world_cups.csv` in the same directory as the notebook
3. Launch JupyterLab or Jupyter Notebook
4. Run all cells in order — each step builds on the previous one

```bash
git clone https://github.com/your-username/worldcup-regression-analysis.git
cd worldcup-regression-analysis
jupyter lab WorldCup_Regression.ipynb
```

---

## Skills Demonstrated

- Exploratory Data Analysis with Matplotlib and Seaborn
- Feature engineering from raw match and tournament data
- Ordinary Least Squares regression with statsmodels
- Variance Inflation Factor calculation and interpretation
- Eigenvalue analysis of the design matrix
- Ridge regression with cross-validated hyperparameter selection
- Principal Component Analysis for dimensionality and collinearity reduction
- Systematic model comparison and diagnostic visualisation

---

## Academic Context

**Course:** Data Science — B.Tech Computer Science Engineering
**Institution:** Sardar Patel Institute of Technology, Mumbai
**Year:** 2025

---

## Author

**Soham Haridas Gaikwad**
B.Tech CSE, SPIT Mumbai — Class of 2028

GitHub: [github.com/your-username](https://github.com/your-username)
LinkedIn: [linkedin.com/in/your-username](https://linkedin.com/in/your-username)
