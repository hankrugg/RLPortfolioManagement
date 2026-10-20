import numpy as np


EXPERIMENT_CONFIG = {
    "experiment_name": "long_short_residual_actions",
    "train_model": True,
    "data": {
        "feature_path": "../data/processed/market_features.csv.gz",
        "development_train_fraction": 0.80,
        "final_holdout_years": 2,
        "scaling_window_years": 5,
        "tradable_symbols": [
            "XLB", "XLE", "XLF", "XLI", "XLK",
            "XLP", "XLU", "XLV", "XLY",
        ],
        "asset_features": [
            "return_5d",
            "return_21d",
            "return_63d",
            "volatility_21d",
            "rolling_drawdown_63d",
            "cross_sectional_rank_return_21d",
            "market_relative_return_21d",
            "rolling_market_beta_63d",
        ],
        "spy_features": [
            "return_5d",
            "return_21d",
            "return_63d",
            "volatility_21d",
            "rolling_drawdown_63d",
        ],
        "market_features": [
            "sector_breadth_above_sma_63d",
            "sector_return_dispersion_21d",
            "sector_average_correlation_63d",
            "vix_level",
            "vix_return_5d",
        ],
    },
    "environment": {
        "decision_frequency_days": 5,
        "action_type": "delta",
        "transaction_cost_rate": 0.0005,
        "annual_short_borrow_rate": 0.01,
        "maximum_gross_exposure": 1.60,
        "maximum_turnover_per_decision": 0.25,
        "observation_clip": 10.0,
    },
    "reward": {
        "name": "after_cost_log_growth",
        "scale": 100.0,
    },
    "training": {
        "algorithm": "PPO",
        "seed": 42,
        "timesteps": 300_000,
        "episode_years": 5,
        "learning_rate": 3e-4,
        "ppo_steps": 2_048,
        "batch_size": 64,
    },
}


def action_to_weights(action, current_weights):
    """Apply a net-zero portfolio change subject to turnover and leverage limits."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    current_weights = np.asarray(current_weights, dtype=np.float64).reshape(-1)
    turnover_limit = EXPERIMENT_CONFIG["environment"][
        "maximum_turnover_per_decision"
    ]
    gross_limit = EXPERIMENT_CONFIG["environment"][
        "maximum_gross_exposure"
    ]

    base_weights = current_weights.copy()
    if np.abs(base_weights).sum() > gross_limit:
        long_exposure = (gross_limit + 1) / 2
        short_exposure = (gross_limit - 1) / 2
        long_weights = np.maximum(base_weights, 0)
        short_weights = np.maximum(-base_weights, 0)
        base_weights = (
            long_exposure * long_weights / long_weights.sum()
            - short_exposure * short_weights / short_weights.sum()
        )

    deleveraging_turnover = np.abs(base_weights - current_weights).sum()
    if deleveraging_turnover >= turnover_limit:
        scale = turnover_limit / deleveraging_turnover
        return (
            current_weights + scale * (base_weights - current_weights)
        ).astype(np.float32)

    remaining_turnover = turnover_limit - deleveraging_turnover
    weight_change = action - action.mean()
    weight_change *= remaining_turnover / max(
        np.abs(weight_change).sum(), remaining_turnover
    )
    weights = base_weights + weight_change

    if np.abs(weights).sum() > gross_limit:
        lower_scale, upper_scale = 0.0, 1.0
        for _ in range(30):
            scale = (lower_scale + upper_scale) / 2
            trial_weights = base_weights + scale * weight_change
            if np.abs(trial_weights).sum() <= gross_limit:
                lower_scale = scale
            else:
                upper_scale = scale
        weights = base_weights + lower_scale * weight_change

    return weights.astype(np.float32)


def calculate_reward(net_return, **portfolio_state):
    """After-cost log portfolio growth."""
    reward_scale = EXPERIMENT_CONFIG["reward"]["scale"]
    return float(reward_scale * np.log1p(net_return))
