
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class SectorAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset,
        episode_length=None,
        random_start=False,
        *,
        observation_clip=10.0,
        decision_frequency_days=5,
        action_type="delta",
        action_scale=0.1,
        long_exposure=1.30,
        short_exposure=0.30,
        maximum_turnover_per_decision=None,
        transaction_cost_rate=0.0005,
        annual_short_borrow_rate=0.01,
        reward_scale=100.0,
        trading_days_per_year=252,
    ):
        super().__init__()
        self.dates = dataset["dates"]
        self.asset_states = dataset["asset_states"]
        self.spy_states = dataset["spy_states"]
        self.market_states = dataset["market_states"]
        self.asset_returns = dataset["asset_returns"]
        self.n_assets = self.asset_returns.shape[1]
        self.episode_length = episode_length or len(self.dates) - 1
        self.random_start = random_start
        self.observation_clip = observation_clip
        self.decision_frequency_days = decision_frequency_days
        self.action_type = action_type
        self.action_scale = action_scale
        self.long_exposure = long_exposure
        self.short_exposure = short_exposure
        self.maximum_turnover_per_decision = maximum_turnover_per_decision
        self.transaction_cost_rate = transaction_cost_rate
        self.annual_short_borrow_rate = annual_short_borrow_rate
        self.reward_scale = reward_scale
        self.trading_days_per_year = trading_days_per_year

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        observation_size = (
            self.asset_states.shape[1]
            + self.spy_states.shape[1]
            + self.market_states.shape[1]
            + self.n_assets
        )
        self.observation_space = spaces.Box(
            low=-self.observation_clip,
            high=self.observation_clip,
            shape=(observation_size,),
            dtype=np.float32,
        )

    def observation(self):
        return np.concatenate(
            [
                self.asset_states[self.current_index],
                self.spy_states[self.current_index],
                self.market_states[self.current_index],
                self.weights,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        latest_start = len(self.dates) - self.episode_length - 1
        self.current_index = (
            int(self.np_random.integers(latest_start + 1))
            if self.random_start and latest_start > 0
            else 0
        )
        self.end_index = self.current_index + self.episode_length
        self.weights = np.full(
            self.n_assets, 1 / self.n_assets, dtype=np.float32
        )
        self.portfolio_value = 1.0
        return self.observation(), {}

    def step(self, action):
        decision_index = self.current_index
        target_weights = action_to_weights(
            action,
            self.weights,
            action_scale=self.action_scale,
            long_exposure=self.long_exposure,
            short_exposure=self.short_exposure,
            action_type=self.action_type,
        )
        if (
            self.action_type == "target"
            and self.maximum_turnover_per_decision is not None
        ):
            weight_change = target_weights - self.weights
            requested_turnover = np.abs(weight_change).sum()
            turnover_scale = self.maximum_turnover_per_decision / max(
                requested_turnover, self.maximum_turnover_per_decision
            )
            target_weights = self.weights + turnover_scale * weight_change
        period_end = min(
            decision_index + self.decision_frequency_days, self.end_index
        )
        total_reward = 0.0
        daily_records = []

        for next_index in range(decision_index + 1, period_end + 1):
            rebalance = next_index == decision_index + 1
            day_weights = (
                target_weights if rebalance else self.weights
            )
            turnover = float(np.abs(day_weights - self.weights).sum())
            result = calculate_return_components(
                day_weights,
                self.asset_returns[next_index],
                turnover,
                transaction_cost_rate=self.transaction_cost_rate,
                annual_short_borrow_rate=self.annual_short_borrow_rate,
                trading_days_per_year=self.trading_days_per_year,
            )
            next_portfolio_value = (
                self.portfolio_value * (1 + result["net_return"])
            )

            reward = calculate_reward(
                net_return=result["net_return"],
                reward_scale=self.reward_scale,
            )
            result["reward"] = reward
            total_reward += reward
            self.portfolio_value = next_portfolio_value

            self.weights = drift_weights(
                day_weights, self.asset_returns[next_index]
            )

            self.current_index = next_index
            daily_records.append(
                {
                    **result,
                    "date": pd.Timestamp(self.dates[next_index]),
                    "turnover": turnover,
                    "portfolio_value": self.portfolio_value,
                    "executed_weights": day_weights.copy(),
                }
            )

        truncated = self.current_index >= self.end_index
        return (
            self.observation(),
            total_reward,
            False,
            truncated,
            {"daily_records": daily_records},
        )



def build_feature_panel(data, symbols, features):
    """Place each symbol's features in columns indexed by date."""
    return pd.concat(
        {
            symbol: (
                data.loc[data["symbol"] == symbol]
                .set_index("date")[features]
                .sort_index()
            )
            for symbol in symbols
        },
        axis=1,
    )


def scale_rolling(panel, window, observation_clip):
    """Standardize each observation using only preceding observations."""
    history = panel.shift(1)
    mean = history.rolling(window).mean()
    standard_deviation = history.rolling(window).std(ddof=0)
    return ((panel - mean) / standard_deviation).clip(
        -observation_clip, observation_clip # Numerical stability and to avoid extreme values that could destabilize training, not intended to exclude outliers
    )


def slice_dataset(dataset, mask):
    """Select one chronological period from every dataset array."""
    return {name: values[mask] for name, values in dataset.items()}


def drift_weights(weights, asset_returns):
    """Update portfolio weights after one day of asset returns."""
    portfolio_return = weights @ asset_returns
    return (
        weights * (1 + asset_returns) / (1 + portfolio_return)
    ).astype(np.float32)


def calculate_return_components(
    weights,
    asset_returns,
    turnover,
    transaction_cost_rate,
    annual_short_borrow_rate,
    *,
    trading_days_per_year=252,
):
    """Calculate gross return, implementation costs, and net return."""
    gross_return = float(weights @ asset_returns)
    transaction_cost = transaction_cost_rate * turnover
    short_exposure = float(np.maximum(-weights, 0).sum())
    short_borrow_cost = (
        annual_short_borrow_rate / trading_days_per_year * short_exposure
    )
    net_return = gross_return - transaction_cost - short_borrow_cost
    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "transaction_cost": transaction_cost,
        "short_borrow_cost": short_borrow_cost,
        "gross_exposure": float(np.abs(weights).sum()),
        "short_exposure": short_exposure,
    }

def action_to_weights(
    action,
    current_weights,
    action_scale,
    long_exposure,
    short_exposure,
    *,
    action_type="delta",
):
    """Apply a residual action with long-only or long/short exposure limits."""
    if action_type != "delta":
        raise ValueError("action_to_weights currently supports only 'delta' actions.")
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    current_weights = np.asarray(current_weights, dtype=np.float64).reshape(-1)
    weight_change = action_scale * (action - action.mean())
    weights = current_weights + weight_change

    if short_exposure == 0:
        # Reuse the simplex projection from the original long-only experiment.
        if long_exposure <= 0:
            raise ValueError("Long-only allocation requires positive long_exposure.")
        sorted_weights = np.sort(weights)[::-1]
        cumulative_weights = np.cumsum(sorted_weights) - long_exposure
        positions = np.arange(1, len(weights) + 1)
        active_position = np.nonzero(
            sorted_weights - cumulative_weights / positions > 0
        )[0][-1]
        threshold = cumulative_weights[active_position] / (active_position + 1)
        return np.maximum(weights - threshold, 0).astype(np.float32)

    if np.abs(weights).sum() > long_exposure + short_exposure:
        long_weights = np.maximum(weights, 0)
        short_weights = np.maximum(-weights, 0)
        weights = (
            long_exposure * long_weights / long_weights.sum()
            - short_exposure * short_weights / short_weights.sum()
        )

    return weights.astype(np.float32)


def calculate_reward(net_return, reward_scale):
    """After-cost log portfolio growth."""
    return float(reward_scale * np.log1p(net_return))
