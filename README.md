#Closed Loop Bias Correction for Predict Then Optimise Shared Battery Dispatch in Apartment Energy

Code accompanying the paper *"Closed Loop Bias Correction for Predict Then Optimise Shared Battery Dispatch in Apartment Energy."*

The framework adapts a frozen, population-trained demand-forecasting model to a
target apartment energy-sharing scheme with a closed-loop, block-resolved
multiplicative (PI-style) bias-correction law, then drives a mixed-integer-linear
(MILP) battery-dispatch model evaluated on **realised settlement cost** in a
full-year rolling-horizon MPC.

## Repository structure

| Path | Role (paper section) |
|------|----------------------|
| `xg_boost_local/` | Per-unit XGBoost demand-forecasting model — feature engineering and training (§7.3) |
| `svr_local/` | Scalable SVR forecasting baselines: LinearSVR and Nyström-RBF + LinearSVR (§8.4) |
| `optimising model/` | Closed-loop PI bias correction, MILP dispatch, rolling-horizon MPC, evaluation, and figures (§5–§9) |

Key modules: `*/config.py` hold paths and hyperparameters; `*/train.py` train the
forecasters; `optimising model/main.py` runs the MPC + MILP and computes realised
cost; `optimising model/sweep_pi.py` runs the PI hyperparameter sweep; the
`plot_*.py` scripts reproduce the figures.

## Data

The study uses publicly available data obtained from its custodians (the raw data
is **not** redistributed here; see the licences/terms of each source):

- **SGSC Ausgrid smart-meter load profiles** — Smart Grid Smart City trial,
  pre-processed Mendeley release (Roberts et al.).
- **Bureau of Meteorology** daily weather, Station 066212 (Sydney Observatory Hill).
- **SGSC household demographic survey**.

Place the downloaded data where each module's `config.py` expects it. Derived
feature matrices can be made available by the corresponding author on reasonable
request.

## Requirements

Python 3 with `xgboost`, `scikit-learn`, `gurobipy`, `pandas`, `numpy`, and
`matplotlib`. The MILP is solved with **Gurobi**, which requires a licence (free
academic licences are available).

## Reproducing the results

Paths and hyperparameters are configured in each module's `config.py`. The
pipeline runs in three stages:

1. **Train the forecaster** — `python xg_boost_local/train.py` (XGBoost) and/or
   `python svr_local/train.py` (SVR baselines).
2. **Run the dispatch + evaluation** — `python "optimising model/main.py"`
   (rolling-horizon MPC, MILP dispatch, realised settlement cost).
3. **PI sweep / ablation and figures** — `python "optimising model/sweep_pi.py"`,
   then the `plot_*.py` scripts.

## Licence

Code released under the [MIT License](LICENSE). Please cite the paper if you use
this code.
