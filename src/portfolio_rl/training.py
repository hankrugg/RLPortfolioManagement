import shutil

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .environment import SectorAllocationEnv


def train_agent(
    train_data,
    split_summary,
    model_directory,
    config_path,
    seed,
    train_episode_years,
    learning_rate,
    ppo_steps,
    batch_size,
    training_timesteps,
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
    device="cpu",
):
    model_path = model_directory / "model.zip"
    model_directory.mkdir(parents=True, exist_ok=True)
    saved_config = model_directory / f"experiment_config{config_path.suffix}"
    if config_path.resolve() != saved_config.resolve():
        shutil.copy2(config_path, saved_config)
    split_summary.to_csv(model_directory / "data_splits.csv", index=False)

    training_episode_length = min(
        train_episode_years * 252, len(train_data["dates"]) - 1
    )
    training_env = Monitor(
        SectorAllocationEnv(
            train_data,
            episode_length=training_episode_length,
            random_start=True,
            observation_clip=observation_clip,
            decision_frequency_days=decision_frequency_days,
            action_type=action_type,
            action_scale=action_scale,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            maximum_turnover_per_decision=maximum_turnover_per_decision,
            transaction_cost_rate=transaction_cost_rate,
            annual_short_borrow_rate=annual_short_borrow_rate,
            reward_scale=reward_scale,
            trading_days_per_year=trading_days_per_year,
        )
    )
    model = PPO(
        "MlpPolicy",
        training_env,
        learning_rate=learning_rate,
        n_steps=ppo_steps,
        batch_size=batch_size,
        seed=seed,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=training_timesteps)
    model.save(model_path)
    training_env.close()
    return model


def evaluate_agent(
    trained_model,
    validation_data,
    tradable_symbols,
    seed,
    model_directory,
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
    validation_environment = SectorAllocationEnv(
        validation_data,
        observation_clip=observation_clip,
        decision_frequency_days=decision_frequency_days,
        action_type=action_type,
        action_scale=action_scale,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        maximum_turnover_per_decision=maximum_turnover_per_decision,
        transaction_cost_rate=transaction_cost_rate,
        annual_short_borrow_rate=annual_short_borrow_rate,
        reward_scale=reward_scale,
        trading_days_per_year=trading_days_per_year,
    )
    observation, _ = validation_environment.reset(seed=seed)
    validation_records = []
    done = False

    while not done:
        action, _ = trained_model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = (
            validation_environment.step(action)
        )

        for daily_result in info["daily_records"]:
            weights = daily_result.pop("executed_weights")
            daily_result.update(
                {
                    f"weight_{symbol}": float(weight)
                    for symbol, weight in zip(tradable_symbols, weights)
                }
            )
            validation_records.append(daily_result)

        done = terminated or truncated

    validation_environment.close()
    validation_results = pd.DataFrame(validation_records).set_index("date")
    validation_results.drop(columns="reward").to_csv(
        model_directory / "validation_results.csv", index_label="date"
    )

    returns = validation_results["net_return"]
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    validation_metrics = pd.Series(
        {
            "cumulative net return": cumulative_return,
            "annualized net return": annualized_return,
            "annualized volatility": annualized_volatility,
            "Sharpe ratio": returns.mean() / returns.std() * np.sqrt(252),
            "maximum drawdown": drawdown.min(),
            "annualized turnover": validation_results["turnover"].mean() * 252,
            "average gross exposure": validation_results["gross_exposure"].mean(),
            "maximum gross exposure": validation_results["gross_exposure"].max(),
            "average short exposure": validation_results["short_exposure"].mean(),
            "annualized implementation cost": (
                validation_results["transaction_cost"]
                + validation_results["short_borrow_cost"]
            ).mean() * 252,
        },
        name="PPO agent",
    )
    validation_metrics.to_csv(
        model_directory / "validation_metrics.csv", header=["value"]
    )
    return validation_results, validation_metrics
