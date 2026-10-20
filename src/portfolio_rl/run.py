"""Run a YAML batch as independent training/evaluation processes."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = PROJECT_ROOT / "configs" / "batch.yaml"


def run_one(config, seed, source, run_directory, threads_per_worker):
    """Train and evaluate one config/seed/source combination in a worker."""
    with (run_directory / "run.log").open("w") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                import torch

                from .data import prepare_datasets
                from .training import evaluate_agent, train_agent

                torch.set_num_threads(threads_per_worker)
                config = deepcopy(config)
                config["training"]["seed"] = seed
                config["training"].pop("seeds", None)
                config["data"]["feature_path"] = source["path"]
                config["data_source"] = source
                config.pop("prepared_dataset", None)
                config_path = run_directory / "experiment_config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))

                # A failed rerun must not leave old returns looking like new results.
                (run_directory / "validation_results.csv").unlink(missing_ok=True)
                train_data, validation_data, split_summary = prepare_datasets(
                    **config["data"],
                    observation_clip=config["environment"].get("observation_clip", 10.0),
                )
                training = config["training"]
                environment = dict(config["environment"])
                environment["reward_scale"] = config["reward"]["scale"]
                model = train_agent(
                    train_data, split_summary, run_directory, config_path, seed,
                    train_episode_years=training["episode_years"],
                    learning_rate=training["learning_rate"],
                    ppo_steps=training["ppo_steps"],
                    batch_size=training["batch_size"],
                    training_timesteps=training["timesteps"],
                    device=training.get("device", "cpu"),
                    **environment,
                )
                evaluate_agent(
                    model, validation_data, config["data"]["tradable_symbols"],
                    seed, run_directory, **environment,
                )
                return {"status": "completed"}
            except Exception as error:
                traceback.print_exc()
                return {"status": "failed", "error": str(error)}


def main(batch_path=None):
    batch_path = PROJECT_ROOT / Path(batch_path) if batch_path else DEFAULT_BATCH
    batch = yaml.safe_load(batch_path.read_text())
    seeds = batch["seeds"]
    sources = deepcopy(batch["data_sources"])
    workers = batch.get("max_workers", 2)
    threads = batch.get("threads_per_worker", 1)
    if type(workers) is not int or workers < 1 or type(threads) is not int or threads < 1:
        raise ValueError("max_workers and threads_per_worker must be positive integers.")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Provide a nonempty list of integer seeds between 0 and 2**32 - 1.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique within a batch.")
    if not sources or len({source["name"] for source in sources}) != len(sources):
        raise ValueError("Provide data sources with unique names.")
    for source in sources:
        name = source["name"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Data source names must be simple folder names.")
        source["path"] = str((PROJECT_ROOT / source["path"]).resolve())
        if not Path(source["path"]).is_file():
            raise FileNotFoundError(source["path"])

    configs = [yaml.safe_load((PROJECT_ROOT / path).read_text())
               for path in batch["experiment_configs"]]
    names = [config["experiment_name"] for config in configs]
    if not configs or len(set(names)) != len(names):
        raise ValueError("Provide configs with unique experiment_name values.")
    for config in configs:
        name = config["experiment_name"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Experiment names must be simple folder names.")
        if config["training"].get("algorithm", "PPO") != "PPO":
            raise ValueError("The current training function supports PPO only.")
        if config["reward"].get("name", "after_cost_log_growth") != "after_cost_log_growth":
            raise ValueError("The current reward is after_cost_log_growth.")

    # Children inherit these limits before importing numerical libraries.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)

    output_directory = PROJECT_ROOT / batch.get("output_directory", "results")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "batch.yaml").write_text(yaml.safe_dump(batch, sort_keys=False))
    jobs = []
    for config in configs:
        experiment = config["experiment_name"]
        for source in sources:
            for seed in seeds:
                run_directory = output_directory / experiment / source["name"] / f"seed_{seed}"
                run_directory.mkdir(parents=True, exist_ok=True)
                jobs.append((config, seed, source, run_directory, threads))

    failed = False
    print(f"Running {len(jobs)} jobs with up to {workers} workers. Results: {output_directory}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, *job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"status": "failed", "error": str(error)}
                with (jobs[index][3] / "run.log").open("a") as log:
                    log.write(traceback.format_exc())
            failed = failed or result["status"] == "failed"
            config, seed, source, _, _ = jobs[index]
            print(f"{result['status']}: {config['experiment_name']} / {source['name']} / seed {seed}", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
