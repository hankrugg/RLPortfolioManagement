# RL Portfolio Management

Train and evaluate a reinforcement learning agent that allocates a portfolio across nine U.S. sector ETFs. The project uses PPO from Stable-Baselines3, daily market data from Yahoo Finance, and configurable trading and short-borrowing costs. Saved model results can be compared with SPY buy-and-hold and an equal-weight sector portfolio.

## 1. Download the project

Download and extract the repository ZIP, or clone it:

```bash
git clone https://github.com/hankrugg/RLPortfolioManagement.git
cd RLPortfolioManagement
```

If you downloaded a ZIP, open a terminal in the extracted project folder, where `requirements.txt` is located. Git is only needed for the clone option.

## 2. Install the dependencies

Use **Python 3.11**, the version specified in `requirements.txt`. The commands below use Conda, available through Miniconda or Anaconda, to create a separate environment. They work in a terminal on macOS/Linux or an Anaconda Prompt on Windows.

```bash
conda create -n rl-portfolio python=3.11 -y
conda activate rl-portfolio
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rl-portfolio --display-name "Python (rl-portfolio)"
```

If you already have an isolated Python 3.11 environment, activate it and run the three `python -m ...` installation commands above.

The requirements file pins NumPy, pandas, yfinance, Gymnasium, Stable-Baselines3, PyTorch, Matplotlib, JupyterLab, and ipykernel. An internet connection is needed to install packages and download market data.

## 3. Start JupyterLab

From the project folder, with the environment activated:

```bash
cd notebooks
python -m jupyterlab
```

Open the notebooks in JupyterLab and select **Python (rl-portfolio)** as the kernel. In each notebook, run all cells from top to bottom and wait for it to finish before moving to the next notebook.

**Keep `notebooks/` as the working directory.** Paths such as `../data/` and `../models/` are relative to this folder, so the project can be downloaded anywhere without editing machine-specific paths. If using VS Code or another notebook editor, make sure its notebook working directory is also `notebooks/`.

## 4. Run the notebooks in order

| Order | Notebook | What it does | Main output |
| --- | --- | --- | --- |
| 1 | [01_download_data.ipynb](notebooks/01_download_data.ipynb) | Downloads adjusted daily prices for the nine sector ETFs, SPY, and VIX. | `data/raw/yahoo_daily_adjusted.csv.gz` |
| 2 | [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) | Builds the price, relative-performance, and market-regime features. | `data/processed/market_features.csv.gz` |
| 3 | [03_train_portfolio_agent_daily.ipynb](notebooks/03_train_portfolio_agent_daily.ipynb) | Trains one PPO agent per configured seed, then evaluates each on development validation. | `models/<experiment_name>/seed_<seed>/` and a validation summary |
| 4 | [04_benchmark_analysis.ipynb](notebooks/04_benchmark_analysis.ipynb) | Compares saved validation results against SPY and an equal-weight sector portfolio. | Performance tables and plots displayed in the notebook |

The `data/` folder is excluded from Git and is **not included in a fresh download**. Run notebooks 01 and 02 to generate it; the notebooks create their output directories as needed.

Notebook 01 currently requests data from `1999-01-01` through an exclusive end date of `2026-10-01`. Keep these settings when comparing with the included model results. Changing the date range can change the training/validation splits and make existing saved results incompatible with the new data.

### Training settings

Before running notebook 03, review [notebooks/experiment_config.py](notebooks/experiment_config.py). It controls the experiment name, whether to train, seeds, training duration, portfolio constraints, costs, and reward function.

The current defaults are:

```python
"experiment_name": "long_short_scaled"
"train_model": True
# Inside "training":
"seeds": [42]
"timesteps": 300_000
```

Training takes longer than data preparation and analysis; runtime depends on your hardware and the number of seeds. Reducing `training.timesteps` can help check the workflow, but produces a different model.

With `train_model=True`, notebook 03 saves checkpoints and validation files under the configured experiment and seed, replacing existing files at those locations. Use a new `experiment_name` to preserve the included models. Each seed folder contains:

- `model.zip`: the PPO checkpoint.
- `experiment_config.py`: the configuration and portfolio/reward functions used for training.
- `data_splits.csv`: the chronological dataset boundaries.
- `validation_results.csv`: daily validation returns, costs, exposures, and weights.
- `validation_metrics.csv`: summary validation metrics.

To evaluate an existing checkpoint without retraining, set `train_model=False` and use the experiment name, seeds, and configuration matching that checkpoint. Validation files are still regenerated. Restart the kernel and run all cells after changing `experiment_config.py`, so Python loads the updated settings.

### Compare the included results without retraining

For a faster first run, complete **01 → 02 → 04**, skipping notebook 03. The repository includes saved validation results for `long_only`, `long_short`, and `long_short_scaled`.

Notebook 04 selects strategies through `MODELS_TO_COMPARE`. It reads each model's saved CSV files directly, looking in the model folder first and then in `seed_42/`. It also uses `models/long_only/validation_results.csv` for the benchmark dates. Keep that file available and compare models evaluated on the same dates.

If you train a new experiment with seed 42, add its experiment name to `MODELS_TO_COMPARE`. For another seed, adjust the model-loading path in notebook 04 to the corresponding `seed_<seed>/` folder.

### Optional exploratory notebook

If your download includes `06_sector_stock_fund_preliminary.ipynb`, run it after notebooks 01 and 02 and after ensuring the `long_short_scaled/seed_42` checkpoint and its saved configuration are available. It reuses the frozen allocator with stock baskets and downloads additional Yahoo Finance data. It is separate from the main 01–04 workflow; no notebook 05 is required.

## Project layout

```text
RLPortfolioManagement/
├── requirements.txt
├── notebooks/            # Run the notebooks from this folder
│   ├── 01_download_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_train_portfolio_agent_daily.ipynb
│   ├── 04_benchmark_analysis.ipynb
│   └── experiment_config.py
├── data/                 # Generated locally; excluded from Git
│   ├── raw/
│   └── processed/
├── models/               # Checkpoints, configurations, and validation CSVs
└── docs/                 # Project report
```

## Troubleshooting

- **`ModuleNotFoundError`:** Select the `Python (rl-portfolio)` kernel. Install dependencies with the same environment activated, then restart the kernel.
- **Missing raw or feature data:** Run notebook 01, then 02. Confirm the notebook working directory is `notebooks/`; `Path.cwd()` in a notebook cell shows it.
- **Missing model or validation files:** Check the experiment name and seed. Train the checkpoint with notebook 03, or select one of the included experiments in notebook 04.
- **Yahoo Finance download errors or missing symbols:** Check your connection and retry notebook 01 after a pause. Wait for a complete dataset before continuing to feature engineering.
- **Dependency installation fails on a different Python version:** Create a fresh Python 3.11 environment and install `requirements.txt` again.

Saved notebook outputs show earlier runs; rerun the cells to generate results for your environment. The comparisons use development validation, while the final two-year holdout is reserved by the training workflow. Fresh downloads and retraining may produce different results from the saved outputs.
