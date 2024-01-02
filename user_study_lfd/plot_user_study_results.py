import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ez_tuning import ResultReporter, bootstrapped_iqm, iqm

matplotlib.use("Agg")
sns.set_theme(context="paper")
colors = sns.color_palette()
fig, axes = plt.subplots(ncols=3, constrained_layout=True)
fig: plt.Figure

reporter1 = ResultReporter.load_checkpoint("rerun.ckpt")
reporter2 = ResultReporter.load_checkpoint("other_baselines.ckpt")
reporter3 = ResultReporter.load_checkpoint("bc_pretraining.ckpt")

for i, difficulty, title in zip(
    range(5),
    ("easy", "medium_1", "hard_1"),
    ("Easy", "Medium", "Hard"),
):
    ax: plt.Axes = axes[i]
    ax.set_title(title)

    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    science_keys = set(
        config.science_key
        for config in itertools.chain(reporter1.results.keys(), reporter2.results.keys(), reporter3.results.keys())
        if config.task_key[0] == difficulty
    )
    # for science_key in science_keys:
    for jj, method, method_name, reporter in zip(
        range(5),
        ("RL", "no-comm LFD", "with comm", "no comm"),
        ("MARL", "NC MA-GAIL", "MA-GAIL", "MixTURE"),
        (reporter2, reporter2, reporter1, reporter1),
    ):
        science_key = (method,)
        nuisance_scores = defaultdict(list)

        for config, result in reporter.results.items():
            if config.task_key[0] != difficulty or config.science_key != science_key:
                continue
            nuisance_scores[config.nuisance_key].append(
                result.data[reporter.comparison_metric][-1]
            )
        if not nuisance_scores:
            print(difficulty, method)
            continue

        nuisance_scores = {k: iqm(v) for k, v in nuisance_scores.items()}
        best_nuisance = (max if reporter.mode == "max" else min)(
            nuisance_scores.keys(),
            key=lambda k: nuisance_scores[k],
        )

        best_nuisance_results = []
        for config, result in reporter.results.items():
            if (
                config.task_key[0] == difficulty
                and config.science_key == science_key
                and config.nuisance_key == best_nuisance
            ):
                best_nuisance_results.append(result.data["episode_len"])

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

    bc_result = [v for k, v in reporter3.results.items() if k.task_key[0] == difficulty][0]
    ax.plot(
        np.arange(26) * 40960,
        np.full(26, fill_value=bc_result.data["episode_len"][-1]),
        color=colors[4],
        label="BC+DC" if i == 0 else None,
        linestyle="dashed",
    )

fig.legend(bbox_to_anchor=(0.5, 0), loc="upper center", ncols=5)
fig.set_size_inches(10, 4)
fig.savefig("user_study_results.png", bbox_inches="tight")
