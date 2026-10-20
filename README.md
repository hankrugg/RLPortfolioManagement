# RL Portfolio Management

Train and evaluate PPO agents that allocate a portfolio across nine U.S. sector ETFs. Experiments use YAML configurations, multiple training seeds, and parallel worker processes. The results notebook compares model averages with SPY buy-and-hold and an equal-weight sector portfolio.

## Set up in VS Code

1. Clone or download this repository, then use **File → Open Folder** in VS Code to open `RLPortfolioManagement`, the folder containing `run.py` and `requirements.txt`.
2. Install Microsoft's **Python** and **Jupyter** extensions in VS Code.
3. Install Miniconda or Anaconda if you do not already have Conda. Open **Terminal → New Terminal** and create a Python 3.11 environment:

```bash
conda create -n rl-portfolio python=3.11 -y
conda activate rl-portfolio
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rl-portfolio --display-name "Python (rl-portfolio)"
```

If you already have an isolated Python 3.11 environment, activate it and run the three `python -m ...` commands above. If `conda` is unavailable in the VS Code terminal, run the setup in an Anaconda Prompt or a terminal where Conda works, then reopen VS Code.

4. Open the Command Palette (**Cmd+Shift+P** on macOS or **Ctrl+Shift+P** on Windows/Linux), run **Python: Select Interpreter**, and choose `rl-portfolio`. Open a new terminal and activate that environment before running training.

VS Code documents [interpreter selection](https://code.visualstudio.com/docs/python/environments#_select-an-environment) and [notebook execution](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

## Included data

The processed dataset, `data/processed/market_features.csv.gz`, is included in the repository. You can train directly from this file without downloading market data or running the old preparation notebooks. An internet connection is needed to install dependencies, but the default training batch uses the local dataset.

The features were built from Yahoo Finance data for the sector ETFs, SPY, and VIX. The original download requests dates from `1999-01-01` through an exclusive end date of `2026-10-01`; usable dates depend on feature availability and scaling windows. Each experiment saves its actual training and validation boundaries in `data_splits.csv`.

Only `data/processed/` is included in Git. Raw data, intermediate files, and other data folders remain ignored. Earlier download and feature-engineering notebooks are kept in `archive/notebooks/` for reference.

## Prepare or refresh data

Edit [configs/data.yaml](configs/data.yaml) to set the date range, tradable symbols, and output paths, then run:

```bash
python prepare_data.py
```

The script downloads adjusted daily Yahoo Finance observations, normalizes and validates them, and saves `data/raw/yahoo_daily_adjusted.csv.gz`. It then constructs and validates the features and saves `data/processed/market_features.csv.gz`, the historical input already configured for training. Raw data means normalized market observations; processed data contains unscaled features. Scaling and chronological splits remain part of training preparation.

To use another data configuration or rebuild from a saved raw file without downloading:

```bash
python prepare_data.py configs/data.yaml --raw-file data/raw/yahoo_daily_adjusted.csv.gz
```

Configured paths resolve from the project root, even when the script is launched from another directory. Each output is replaced only after validation and a complete temporary file write succeed. If feature construction fails, the newly validated raw file remains available and the previous processed file is preserved; the command exits with an error. Successful reruns replace the configured outputs.

The script uses historical NYSE sessions (`exchange_calendar: XNYS`) from the pinned `exchange-calendars` dependency. Holidays, weekends, and exceptional closures are excluded explicitly before feature construction, including Yahoo's empty ETF rows on non-session dates. Exclusion counts are logged. A missing required price or observation on an open session raises an error, even if the session is absent from every downloaded series. Early-close days count as sessions. No prices are filled and no missing open-session rows are dropped; expected initial feature warm-up is allowed. Optional `calendar_path` supplies an explicit session CSV with a `date` column instead of the named exchange calendar.

Training validates the processed CSV against the exchange sessions within its date range before preparing arrays. Synthetic datasets can supply an explicit session index; calendar labels must describe the timeline being simulated. The archived notebooks are optional references.

## Configure and run training

Edit [configs/batch.yaml](configs/batch.yaml) to choose experiment configs, seeds, data sources, and concurrency:

```yaml
experiment_configs:
  - configs/long_only.yaml
  - configs/long_short_scaled.yaml
  - configs/long_short_unscaled.yaml

seeds: [42, 123, 2026]

data_sources:
  - name: historical
    path: data/processed/market_features.csv.gz

max_workers: 9
threads_per_worker: 1
output_directory: results
```

The runner trains every config × seed × data source combination. This batch creates nine jobs. `max_workers` sets the maximum number of simultaneous experiments; `threads_per_worker` limits the CPU threads available to each worker. Both must be positive integers. Lower `max_workers` if you want to leave more resources available for other work.

Each experiment YAML controls its name, input features, chronological splits, portfolio settings, reward scale, and PPO training settings. The current configs use 300,000 training steps. For a quick workflow check, reduce `training.timesteps`. All current experiments rebalance every five exchange trading sessions (`environment.decision_frequency_days: 5`). Holidays can shift the weekday and extend calendar time between decisions; early-close sessions still count once. Returns and costs are recorded for every trading session between decisions.

In the VS Code terminal, with the environment activated and the terminal at the **project root**, run:

```bash
python run.py
```

There is no need to switch into `src/` or launch the archived training notebook. The terminal reports each job as it finishes; detailed training output is written to its `run.log`. Keep the process running until the batch finishes.

To use a different batch file:

```bash
python run.py configs/another_batch.yaml
```

Configured relative paths are resolved from the project root. Each worker prepares its inputs in memory, including scaling and chronological splits. The final holdout is excluded from training and development validation.

To add another experiment, copy an experiment YAML, give it a unique `experiment_name`, change the settings, and add its path to `experiment_configs`. The current trainer supports PPO and the `after_cost_log_growth` reward. For alternative or bootstrapped data, add a uniquely named source pointing to a feature CSV with the same schema as the included dataset. Bootstrap generation is separate; the batch seeds control training randomness.

## Saved results

The runner creates the output folders automatically:

```text
results/
├── batch.yaml
└── <experiment>/
    └── <data_source>/
        └── seed_<seed>/
            ├── model.zip
            ├── experiment_config.yaml
            ├── data_splits.csv
            ├── validation_results.csv
            ├── validation_metrics.csv
            └── run.log
```

`validation_results.csv` contains daily evaluation returns, portfolio weights, turnover, costs, and exposures. `validation_metrics.csv` contains that run's metrics. There is no shared summary CSV or prepared-data directory. Generated results are ignored by Git.

Rerunning the same experiment/source/seed replaces its saved outputs. Use a new experiment name or output location if you want to preserve an earlier version. Ordinary job failures are recorded in the job's log while the remaining jobs continue; the runner exits with a nonzero status if any job fails.

## View results in VS Code

1. After training finishes, open [notebooks/results.ipynb](notebooks/results.ipynb).
2. Click **Select Kernel** in the upper-right corner and choose the `rl-portfolio` environment or **Python (rl-portfolio)** kernel.
3. Click **Run All** to load the saved results and regenerate the table and plots.

The notebook discovers `results/<experiment>/<data_source>/seed_*/validation_results.csv` files and reads their saved experiment configs. It calculates summary statistics directly from daily results, then averages them across seeds for each model and data source. New model folders are included automatically when you rerun all cells.

It displays:

- Average summary statistics for each model, plus SPY and equal weight.
- Cumulative returns averaged across seeds, compared with both benchmarks.
- Drawdowns of those average return curves and benchmarks.
- A separate average portfolio weights plot for each model and data source.

Use comparable evaluation dates and asset universes when comparing experiments. The notebook uses the first discovered run to establish benchmark data and settings. If you change the output location in the batch YAML, update `run_directory` near the top of the notebook too. Its default paths work when the notebook's working directory is either the project root or `notebooks/`.

## Project layout

```text
RLPortfolioManagement/
├── run.py                  # Batch entry point
├── prepare_data.py         # Download, validate, and save raw/processed data
├── requirements.txt
├── configs/                # Batch and experiment YAMLs
├── src/portfolio_rl/
│   ├── data.py             # Data preparation and chronological splits
│   ├── calendar.py         # Historical exchange sessions and date selection
│   ├── validation.py       # Market and feature data checks
│   ├── environment.py      # Portfolio environment, actions, and reward
│   ├── training.py         # Training and evaluation
│   └── run.py              # Parallel batch orchestration
├── data/processed/         # Included feature dataset
├── notebooks/results.ipynb # Results table and plots
├── results/                # Generated models and evaluation files; ignored by Git
├── archive/                # Earlier notebooks and models
└── docs/                   # Project report
```

## Troubleshooting

- **Missing Python package:** Activate `rl-portfolio`, install `requirements.txt`, and select the same environment for the VS Code interpreter and notebook kernel. Restart the kernel after installing packages.
- **No results found:** Complete at least one training run and check that the notebook points to the correct output directory.
- **Missing feature data:** Check that `data/processed/market_features.csv.gz` is present and that the batch source path points to it.
- **Training failed:** Open the affected seed folder's `run.log` for the error.
- **Computer becomes sluggish:** Reduce `max_workers`; increasing worker or thread counts does not always improve speed.
- **Dependency installation fails:** Use a fresh Python 3.11 environment, the version used for this project's pinned dependencies.
