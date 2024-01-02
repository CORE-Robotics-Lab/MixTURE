import inspect
import json
import os
import shutil
import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns

from utils import Logger


def iqm(scores: List[float]):
    scores = np.asarray(scores).ravel()

    n_obs = scores.shape[0]
    lowercut = n_obs // 4
    uppercut = n_obs - lowercut

    scores = np.partition(scores, (lowercut, uppercut - 1))
    return np.mean(scores[lowercut:uppercut])


# https://arxiv.org/abs/2108.13264 IQM with bootstrapped confidence intervals, with
# support for NaN results (e.g. incomplete trials)
def bootstrapped_iqm(runs: np.ndarray, iters=1000, alpha=0.95, seed=42):
    assert 0 < alpha < 1

    rng = np.random.default_rng(seed)

    idx = rng.integers(runs.shape[0], size=(iters, runs.shape[0]))
    bootstraps: np.ndarray = runs[idx].astype(float)  # (iters, trials, time)

    lowercut = runs.shape[0] // 4
    uppercut = runs.shape[0] - lowercut

    bootstraps.partition(uppercut - 1, axis=1)
    bootstraps[:, uppercut:] = np.inf

    bootstraps[np.isnan(bootstraps)] = -np.inf
    bootstraps.partition(lowercut, axis=1)
    bootstraps[:, :lowercut] = np.nan

    bootstraps[~np.isfinite(bootstraps)] = np.nan

    # ignore numpy warning: mean of empty slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = np.nanmean(bootstraps, axis=1)

        lo = np.nanquantile(results, 0.5 - alpha / 2, axis=0)
        mid = np.nanmean(results, axis=0)
        hi = np.nanquantile(results, 0.5 + alpha / 2, axis=0)
    return lo, mid, hi


def config_name(config):
    ret = []
    for k, v in config.items():
        if isinstance(v, float):
            ret.append(f"{k}={v:.2g}")
        else:
            ret.append(f"{k}={v}")
    return "_".join(ret)


class TrialConfig:
    def __init__(self, spec, cfg_dict):
        vals_by_type = defaultdict(list)
        for key in cfg_dict:
            assert key in spec, f"Unexpected key: {key}"
        for key in sorted(spec.keys()):
            assert key in cfg_dict, f"{key} not found in config={cfg_dict}"
            assert spec[key] in (
                "task",
                "science",
                "nuisance",
                "id",
            ), f"Unexpected parameter type: {spec[key]}"
            vals_by_type[spec[key]].append(cfg_dict[key])

        self.cfg_dict = cfg_dict

        self.task_key = tuple(vals_by_type["task"])
        self.science_key = tuple(vals_by_type["science"])
        self.nuisance_key = tuple(vals_by_type["nuisance"])
        self.id_key = tuple(vals_by_type["id"])

    def __hash__(self):
        return hash((self.task_key, self.science_key, self.nuisance_key, self.id_key))

    def __eq__(self, other):
        if not isinstance(other, TrialConfig):
            return False

        return (
            self.task_key == other.task_key
            and self.science_key == other.science_key
            and self.nuisance_key == other.nuisance_key
            and self.id_key == other.id_key
        )


class ResultReporter:
    def __init__(self, spec, metric, mode, plot_dir, ckpt_filename):
        self.spec = spec
        self.comparison_metric = metric
        self.mode = mode
        self.results: Dict[TrialConfig, Logger] = {}
        self.finished = set()

        self.task_metrics = sorted(
            [k for k, v in self.spec.items() if v == "task"],
        )
        self.science_metrics = sorted(
            [k for k, v in self.spec.items() if v == "science"],
        )
        self.nuisance_metrics = sorted(
            [k for k, v in self.spec.items() if v == "nuisance"],
        )

        self.cleared_plots = False
        self.plot_dir = plot_dir
        self.ckpt_filename = ckpt_filename

        self.last_ckpt_time = None

    def add_result(
        self, config: TrialConfig, result: Logger, plot=True, checkpoint=True
    ):
        self.results[config] = result

        if self.last_ckpt_time is None or time.time() - self.last_ckpt_time > 30:
            if plot:
                self.plot_results()
            if checkpoint:
                self.checkpoint()
            self.last_ckpt_time = time.time()

    def is_done(self, config: TrialConfig):
        return config in self.finished

    def set_done(self, config: TrialConfig, plot=True, checkpoint=True):
        self.finished.add(config)
        if self.last_ckpt_time is None or time.time() - self.last_ckpt_time > 30:
            if plot:
                self.plot_results()
            if checkpoint:
                self.checkpoint()
            self.last_ckpt_time = time.time()

    def change_spec(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.spec.keys(), f"unexpected key: {k}"
            self.spec[k] = v

        self.task_metrics = sorted(
            [k for k, v in self.spec.items() if v == "task"],
        )
        self.science_metrics = sorted(
            [k for k, v in self.spec.items() if v == "science"],
        )
        self.nuisance_metrics = sorted(
            [k for k, v in self.spec.items() if v == "nuisance"],
        )
        new_finished = set()
        new_results = {}
        for config, result in self.results.items():
            new_config = TrialConfig(self.spec, config.cfg_dict)

            if config in self.finished:
                new_finished.add(new_config)
            new_results[new_config] = result
        self.results = new_results
        self.finished = new_finished

    def plot_results(self, plot_dir=None):
        matplotlib.use("Agg")
        sns.set_theme()

        if not self.results:
            return

        metrics = list(self.results.values())[0].data.keys()
        if plot_dir is None:
            plot_dir = self.plot_dir
        if not self.cleared_plots and os.path.exists(plot_dir):
            self.cleared_plots = True
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        for metric in metrics:
            for task in set(config.task_key for config in self.results.keys()):
                fig, ax = plt.subplots(constrained_layout=True)
                fig: plt.Figure
                ax: plt.Axes

                task_title = ", ".join(
                    f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in zip(self.task_metrics, task)
                )
                filename = str(metric)
                if task_title:
                    title = f"{task_title} :: {metric}"
                    filename = os.path.join(task_title, filename)
                else:
                    title = metric
                filename = os.path.join(plot_dir, filename)

                for config in self.results.keys():
                    if config.task_key == task and config not in self.finished:
                        title += " (in progress)"
                        break

                ax.set_title(title.replace("_", " "))
                ax.set_xlabel("epochs")

                science_keys = set(
                    config.science_key
                    for config in self.results.keys()
                    if config.task_key == task
                )
                for science_key in science_keys:
                    nuisance_scores = defaultdict(list)

                    for config, result in self.results.items():
                        if config.task_key != task or config.science_key != science_key:
                            continue
                        nuisance_scores[config.nuisance_key].append(
                            result.data[self.comparison_metric][-1]
                        )

                    nuisance_scores = {k: iqm(v) for k, v in nuisance_scores.items()}
                    best_nuisance = (max if self.mode == "max" else min)(
                        nuisance_scores.keys(),
                        key=lambda k: nuisance_scores[k],
                    )

                    best_nuisance_results = []
                    for config, result in self.results.items():
                        if (
                            config.task_key == task
                            and config.science_key == science_key
                            and config.nuisance_key == best_nuisance
                        ):
                            best_nuisance_results.append(result.data[metric])

                    max_len = max(len(run) for run in best_nuisance_results)
                    best_nuisance_results = np.array(
                        [
                            run + [np.nan] * (max_len - len(run))
                            for run in best_nuisance_results
                        ]
                    )
                    lo, mid, hi = bootstrapped_iqm(best_nuisance_results)

                    science_label = ", ".join(
                        f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in zip(self.science_metrics, science_key)
                    )
                    nuisance_label = ", ".join(
                        f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in zip(self.nuisance_metrics, best_nuisance)
                    )
                    if science_label and nuisance_label:
                        label = f"{science_label} [{nuisance_label}]"
                    elif science_label:
                        label = science_label
                    elif nuisance_label:
                        label = nuisance_label
                    else:
                        label = None
                    (line,) = ax.plot(
                        mid,
                        label=label,
                    )
                    ax.fill_between(
                        np.arange(len(mid)),
                        lo,
                        hi,
                        color=line.get_color(),
                        alpha=0.3,
                    )

                    if len(mid) <= 100:  # add thick circles for clarity
                        ax.scatter(np.arange(len(mid)), mid, color=line.get_color())

                if self.science_metrics or self.nuisance_metrics:
                    fig.legend(bbox_to_anchor=(1, 1), loc="upper left")

                dirname, _ = os.path.split(filename)
                os.makedirs(dirname, exist_ok=True)
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)

    def checkpoint(self, filename=None):
        configs = list(self.results.keys())
        with open(self.ckpt_filename if filename is None else filename, "w") as f:
            json.dump(
                {
                    "spec": self.spec,
                    "metric": self.comparison_metric,
                    "mode": self.mode,
                    "configs": [cfg.cfg_dict for cfg in configs],
                    "results": [self.results[cfg].data for cfg in configs],
                    "results_std": [self.results[cfg].std_data for cfg in configs],
                    "finished": [cfg in self.finished for cfg in configs],
                },
                f,
            )

    @staticmethod
    def load_checkpoint(filename, plot_dir="tuner_plots", ckpt_filename="tuner.ckpt"):
        with open(filename, "r") as f:
            checkpoint = json.load(f)

        reporter = ResultReporter(
            checkpoint["spec"],
            checkpoint["metric"],
            checkpoint["mode"],
            plot_dir,
            ckpt_filename,
        )
        reporter.results = {}
        for i, cfg_dict in enumerate(checkpoint["configs"]):
            logger = Logger()
            logger.data = checkpoint["results"][i]
            logger.std_data = checkpoint["results_std"][i]

            config = TrialConfig(checkpoint["spec"], cfg_dict)
            reporter.results[config] = logger

            if checkpoint["finished"][i]:
                reporter.finished.add(config)

        return reporter


remote_reporter = ray.remote(ResultReporter)


class Tuner:
    def __init__(
        self,
        spec,
        trial_fn: Callable,
        metric: str,
        mode="max",
        plot_dir="tuner_plots",
        ckpt_filename="tuner.ckpt",
        trial_cpus=1,
        trial_gpus=0,
        throw_on_exception=False,
    ):
        for k, v in spec.items():
            assert v in ("task", "science", "nuisance", "id")
        assert inspect.isgeneratorfunction(trial_fn)
        assert mode in ("min", "max")

        self.spec = spec
        self.metric = metric
        self.mode = mode

        def run_fn(config, reporter):
            if ray.get(reporter.is_done.remote(config)):
                return

            try:
                for result in trial_fn(config.cfg_dict):
                    ray.get(reporter.add_result.remote(config, result))
            except Exception as e:
                print(
                    f"Trial config={config_name(config.cfg_dict)} failed with "
                    f"exception {e}"
                )
                if throw_on_exception:
                    raise e
            finally:
                ray.get(reporter.set_done.remote(config))

        self.reporter = remote_reporter.remote(
            self.spec, self.metric, self.mode, plot_dir, ckpt_filename
        )
        self.run_fn = ray.remote(num_cpus=trial_cpus, num_gpus=trial_gpus)(run_fn)
        self.remote_args = []

    def add(self, cfg_dict):
        self.remote_args.append((TrialConfig(self.spec, cfg_dict), self.reporter))

    def run(self, max_tasks=16):
        result_refs = []
        for args in self.remote_args:
            if len(result_refs) > max_tasks:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
                ray.get(ready_refs)
            result_refs.append(self.run_fn.remote(*args))
        ray.get(result_refs)

        ray.get(self.reporter.plot_results.remote())
        ray.get(self.reporter.checkpoint.remote())

    @staticmethod
    def load_checkpoint(trial_fn, filename, **kwargs):
        reporter = ResultReporter.load_checkpoint(filename)

        tuner = Tuner(
            reporter.spec,
            trial_fn,
            reporter.comparison_metric,
            reporter.mode,
            **kwargs,
        )
        for config, result in reporter.results.items():
            ray.get(
                tuner.reporter.add_result.remote(
                    config, result, plot=False, checkpoint=False
                )
            )
        for config in reporter.finished:
            ray.get(
                tuner.reporter.set_done.remote(config, plot=False, checkpoint=False)
            )

        return tuner
