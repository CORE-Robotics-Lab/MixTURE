from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ez_tuning import ResultReporter, bootstrapped_iqm, iqm

matplotlib.use("Agg")
sns.set_theme(context="paper", palette="deep")
colors = sns.color_palette()
fig, axes = plt.subplots(ncols=3, constrained_layout=True)
fig: plt.Figure

reporter = ResultReporter.load_checkpoint("tuner.ckpt")

for i, env, env_title in zip(
    range(3),
    (
        "predatorprey_5x5",
        "predatorcapture_5x5",
        "firecommander_5x5",
    ),
    (
        "Predator-Prey 5x5",
        "Predator-Capture-Prey 5x5",
        "FireCommander 5x5",
    ),
):
    ax: plt.Axes = axes[i]
    ax.set_title(env_title)

    ax.axhline(0, color="black")
    ax.axvline(0, color="black")

    science_keys = set(
        config.science_key
        for config in reporter.results.keys()
        if config.task_key[0] == env
    )
    # for science_key in science_keys:
    for jj, use_mim, method_name in zip(
        range(3),
        (False, True),
        ("no reconstruction", "with reconstruction"),
    ):
        science_key = (use_mim,)
        nuisance_scores = defaultdict(list)

        for config, result in reporter.results.items():
            if config.task_key[0] != env or config.science_key != science_key:
                continue
            nuisance_scores[config.nuisance_key].append(
                result.data[reporter.comparison_metric][-1]
            )
        if not nuisance_scores:
            print(env, use_mim)
            continue

        nuisance_scores = {k: iqm(v) for k, v in nuisance_scores.items()}
        best_nuisance = (max if reporter.mode == "max" else min)(
            nuisance_scores.keys(),
            key=lambda k: nuisance_scores[k],
        )

        best_nuisance_results = []
        for config, result in reporter.results.items():
            if (
                config.task_key[0] == env
                and config.science_key == science_key
                and config.nuisance_key == best_nuisance
            ):
                best_nuisance_results.append(
                    result.data["episode_len"]
                    + (51 - len(result.data["episode_len"])) * [80]
                )

        max_len = max(len(run) for run in best_nuisance_results)
        best_nuisance_results = np.array(
            [run + [np.nan] * (max_len - len(run)) for run in best_nuisance_results]
        )
        lo, mid, hi = bootstrapped_iqm(best_nuisance_results)

        ax.ticklabel_format(axis="x", scilimits=(0, 0))

        ax.set_xlabel("timesteps")
        if i == 0:
            ax.set_ylabel("episode length")

        ax.plot(
            np.arange(len(mid)) * 40960,
            mid,
            label=method_name if i == 0 else None,
            color=colors[jj],
        )
        ax.fill_between(
            np.arange(len(mid)) * 40960,
            lo,
            hi,
            color=colors[jj],
            alpha=0.3,
        )

fig.legend(bbox_to_anchor=(0.5, 0), loc="upper center", ncols=3)
fig.set_size_inches(10, 4)
fig.savefig("mim_ablation_results.png", bbox_inches="tight")
