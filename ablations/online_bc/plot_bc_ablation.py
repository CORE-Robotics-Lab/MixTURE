from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ez_tuning import ResultReporter, bootstrapped_iqm, iqm

matplotlib.use("Agg")
sns.set_theme(context="paper", palette="deep")
colors = sns.color_palette()

fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
fig: plt.Figure

reporter = ResultReporter.load_checkpoint(f"tuner.ckpt")

for i, difficulty, size in zip(range(3), ("easy", "medium"), (5, 10)):
    for j, task, task_name in zip(
        range(3),
        ("predatorprey", "predatorcapture", "firecommander"),
        ("Predator-Prey", "Predator-Capture-Prey", "FireCommander"),
    ):
        ax: plt.Axes = axes[i, j]
        ax.set_title(f"{task_name} {size}x{size}")

        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        science_keys = set(
            config.science_key
            for config in reporter.results.keys()
            if config.task_key[0] == f"{task}_{size}x{size}"
        )
        # for science_key in science_keys:
        for jj, method, method_name in zip(
            range(3),
            ("none", "offline", "online"),
            ("no BC", "offline BC (pretraining)", "online BC (auxiliary objective)"),
        ):
            science_key = (method,)
            nuisance_scores = defaultdict(list)

            for config, result in reporter.results.items():
                if (
                    config.task_key[0] != f"{task}_{size}x{size}"
                    or config.science_key != science_key
                ):
                    continue
                nuisance_scores[config.nuisance_key].append(
                    result.data[reporter.comparison_metric][-1]
                )
            if not nuisance_scores:
                print(task_name, size, method)
                continue

            nuisance_scores = {k: iqm(v) for k, v in nuisance_scores.items()}
            best_nuisance = (max if reporter.mode == "max" else min)(
                nuisance_scores.keys(),
                key=lambda k: nuisance_scores[k],
            )

            best_nuisance_results = []
            for config, result in reporter.results.items():
                if (
                    config.task_key[0] == f"{task}_{size}x{size}"
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
            if i == 1:
                ax.set_xlabel("timesteps")
            if j == 0:
                ax.set_ylabel("episode length")

            ax.plot(
                np.arange(len(mid)) * 40960,
                mid,
                label=method_name if i == j == 0 else None,
                color=colors[jj],
            )
            ax.fill_between(
                np.arange(len(mid)) * 40960,
                lo,
                hi,
                color=colors[jj],
                alpha=0.3,
            )

fig.legend(bbox_to_anchor=(1, 1), loc="upper left")
fig.set_size_inches(8, 6)
fig.savefig("bc_ablation_results.png", bbox_inches="tight")
