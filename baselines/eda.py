import numpy as np
import pandas as pd

from ez_tuning import ResultReporter

reporter = ResultReporter.load_checkpoint("../ablations/online_bc/tuner.ckpt")
science_metrics = sorted(k for k in reporter.spec if reporter.spec[k] == "science")
nuisance_metrics = sorted(k for k in reporter.spec if reporter.spec[k] == "nuisance")
data = []
for config, logger in reporter.results.items():
    print(config.task_key)
    if config.task_key[0] != "predatorprey_10x10":
        continue
    score_1 = np.min(logger.data["episode_len"])
    score_2 = np.mean(logger.data["episode_len"])
    score_3 = np.percentile(logger.data["episode_len"], 20)

    w = 0.9 ** (
        len(logger.data["episode_len"]) - np.arange(len(logger.data["episode_len"]))
    )
    score_4 = np.average(logger.data["episode_len"], weights=w)

    cfg_str = ", ".join(
        f"{k}: {v:.2g}" for k, v in zip(nuisance_metrics, config.nuisance_key)
    )
    data.append(
        {
            **{k: v for k, v in zip(science_metrics, config.science_key)},
            **{k: v for k, v in zip(nuisance_metrics, config.nuisance_key)},
            **dict(score_1=score_1, score_2=score_2, score_3=score_3, score_4=score_4),
        }
    )
df = pd.DataFrame(data)
df = df.sort_values(by="score_4")
df = df.groupby(by="method", group_keys=True).apply(lambda x: x)

pd.options.display.width = 0
print(df)
