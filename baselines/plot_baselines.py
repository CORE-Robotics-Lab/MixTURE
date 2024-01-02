from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ez_tuning import ResultReporter, bootstrapped_iqm, iqm

matplotlib.use("Agg")
sns.set_theme(context="paper")
colors = sns.color_palette()

fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
fig: plt.Figure

diff_bc_results = {
    (5, "predatorprey"): 11.58,
    (5, "predatorcapture"): 14.93,
    (5, "firecommander"): 38.25,
    (10, "predatorprey"): 19.70,
    (10, "predatorcapture"): 62.43,
    (10, "firecommander"): 52.68,
    (20, "predatorprey"): 44.31,
    (20, "predatorcapture"): 80.00,
    (20, "firecommander"): 73.87,
}
for i, difficulty, size in list(zip(range(3), ("easy", "medium", "hard"), (5, 10, 20))):
    reporter = ResultReporter.load_checkpoint(f"{difficulty}_rerun.ckpt")

    for j, task, task_name in zip(
        range(3),
        ("predatorprey", "predatorcapture", "firecommander"),
        ("Predator-Prey", "Predator-Capture-Prey", "FireCommander"),
    ):
        ax: plt.Axes = axes[i, j]
        # ax: plt.Axes = axes[j]
        ax.set_title(f"{task_name} {size}x{size}")

        # ax.axhline(0, color="black")  # TODO revert
        # ax.axvline(0, color="black")  # TODO revert

        science_keys = set(
            config.science_key
            for config in reporter.results.keys()
            if config.task_key[0] == task
        )
        # for science_key in science_keys:
        for jj, method, method_name in zip(
            range(4),
            ("RL", "no-comm LFD", "comm LFD", "ours"),
            ("MARL", "NC MA-GAIL", "MA-GAIL", "MixTURE"),
        ):
            science_key = (method,)
            nuisance_scores = defaultdict(list)

            for config, result in reporter.results.items():
                if config.task_key[0] != task or config.science_key != science_key:
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
                    config.task_key[0] == task
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
            if i == 2:
                ax.set_xlabel("timesteps")
            if j == 0:
                ax.set_ylabel("episode length")

            ax.plot(
                np.arange(len(mid)) * 40960,
                mid,
                label=method_name if i == 0 and j == 0 else None,
                color=colors[jj],
            )
            ax.fill_between(
                np.arange(len(mid)) * 40960,
                lo,
                hi,
                color=colors[jj],
                alpha=0.3,
            )

        n = 51 if size == 5 else 21
        ax.plot(
            np.arange(n) * 40960,
            np.full(n, fill_value=diff_bc_results[(size, task)]),
            color=colors[4],
            label="BC+DC" if i == 0 and j == 0 else None,
            linestyle="dashed",
        )


fig.legend(bbox_to_anchor=(0.5, 0), loc="upper center", ncols=5)
fig.savefig("baseline_results.png", bbox_inches="tight")
