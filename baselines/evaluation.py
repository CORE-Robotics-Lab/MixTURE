import pickle
import random
from collections import defaultdict

import jax
import numpy as np
import ray
import torch

from agents import (
    FullAttentionMIMAgents,
    RecurrentAgents,
    RecurrentAttentionCommAgents,
    RecurrentMIMAgents2,
)
from behavioral_cloning import BCTrainer
from envs import (
    FireCommander5x5,
    FireCommander10x10,
    FireCommander20x20,
    PredatorCapture5x5,
    PredatorCapture10x10,
    PredatorCapture20x20,
    PredatorPrey5x5,
    PredatorPrey10x10,
    PredatorPrey20x20,
)
from envs.comm_wrapper import add_uniform_comms
from marl import PPOTrainer, encode_samples
from reward_signals import ExplicitReward


def evaluate_performance(agents, env, ckpt_filename):
    trainer = PPOTrainer(env, agents, ExplicitReward())
    agents.load_state_dict(torch.load(ckpt_filename))

    results = []
    for trial in range(10):
        trainer.evaluate(num_episodes=100)
        results.append(trainer.logger.data["episode_len"][-1])

    return np.mean(results), np.std(results)


# returns log likelihood of the joint action
def estimate_log_likelihood(agents, env, ckpt_filename, demo_filename):
    trainer = BCTrainer(env, agents, demo_filename)
    agents.load_state_dict(torch.load(ckpt_filename))

    for _ in range(10):
        trainer.run()

    return -np.mean(trainer.logger.data["loss"]), np.std(trainer.logger.data["loss"])


def joint_obs_visitation(dataset, env):
    if isinstance(dataset["obs"][list(dataset["obs"].keys())[0]], dict):
        dataset["obs"] = {
            agent_id: encode_samples(agent_obs, env.observation_spaces[agent_id])
            for agent_id, agent_obs in dataset["obs"].items()
        }

    counts = defaultdict(int)
    for t in range(len(dataset["terminal"])):
        obs = jax.tree_map(lambda a: a[t], dataset["obs"])
        hsh = ()
        for agent_id in sorted(obs.keys()):
            hsh = hsh + tuple(np.asarray(obs[agent_id]).round(1))
        counts[hsh] += 1
    return counts


def estimate_kl(agents, env, ckpt_filename, demo_filename):
    with open(demo_filename, "rb") as f:
        demo_data = pickle.load(f)

    # D_KL(p(x) || q(x)) = \sum_x p(x) log (p(x) / q(x))
    expert_visitation = joint_obs_visitation(demo_data, env)
    expert_tot = sum(expert_visitation.values())

    trainer = PPOTrainer(
        env,
        agents,
        reward_signal=ExplicitReward(),
        rollout_steps=10000,
    )
    agents.load_state_dict(torch.load(ckpt_filename))

    kls = []
    for _ in range(3):
        agent_visitation = joint_obs_visitation(trainer.collect_rollout(), env)
        # old_len = len(agent_visitation)
        for k in expert_visitation.keys():
            agent_visitation[k] += 1
        agent_tot = sum(agent_visitation.values())

        kl = 0
        for k in agent_visitation.keys():
            p = expert_visitation[k] / expert_tot
            q = agent_visitation[k] / agent_tot
            if p == 0:
                continue
            kl += p * np.log(p / q)
        kls.append(kl)

    return np.mean(kls), np.std(kls)


envs = {
    "predatorprey_5x5": PredatorPrey5x5(),
    "predatorprey_10x10": PredatorPrey10x10(),
    "predatorprey_20x20": PredatorPrey20x20(),
    "predatorcapture_5x5": PredatorCapture5x5(),
    "predatorcapture_10x10": PredatorCapture10x10(),
    "predatorcapture_20x20": PredatorCapture20x20(),
    "firecommander_5x5": FireCommander5x5(),
    "firecommander_10x10": FireCommander10x10(n_fires=1),
    "firecommander_20x20": FireCommander20x20(n_fires=1),
}


def run(env_name, method):
    random.seed(42)
    np.random.seed(random.randrange(2**32))
    torch.manual_seed(random.randrange(2**32))

    env = envs[env_name]

    env_name2 = (
        env_name + "_ez"
        if env_name in ("firecommander_10x10", "firecommander_20x20")
        else env_name
    )

    fc_dim = 64 if env_name.endswith("5x5") else 256
    if method == "RL":
        agents = RecurrentAgents(0, 0, fc_dim=fc_dim)
        ckpt_filename = f"saved_agents/{env_name}/RL_best_0.pth"
        add_comms = True
    elif method == "BC+DC":
        agents = RecurrentAttentionCommAgents(0, 0, fc_dim=fc_dim)
        ckpt_filename = f"pretrained_policies/continuous_bc_{env_name}_best.pth"
        add_comms = False
    elif method == "LFD":
        agents = RecurrentAgents(0, 0, fc_dim=fc_dim)
        ckpt_filename = f"saved_agents/{env_name}/no-comm LFD_best_0.pth"
        add_comms = False
    elif method == "MA-GAIL":
        agents = RecurrentAgents(0, 0, fc_dim=fc_dim)
        ckpt_filename = f"saved_agents/{env_name}/comm LFD_best_0.pth"
        add_comms = True
    else:
        assert method == "MixTURE"
        if fc_dim == 64:
            agents = RecurrentMIMAgents2(0, 0, mim_coeff=0.1, fc_dim=64)
        else:
            agents = FullAttentionMIMAgents(0, 0, mim_coeff=0.01, fc_dim=256)
        add_comms = False
        ckpt_filename = f"saved_agents/{env_name}/ours_best_0.pth"

    if add_comms:
        demo_filename = f"../demos/{env_name2}_comm.pickle"
    else:
        demo_filename = f"../demos/{env_name2}.pickle"

    if add_comms:
        eval_env = add_uniform_comms(env, 26)
    else:
        eval_env = env

    ep_len = evaluate_performance(agents, eval_env, ckpt_filename)
    ll = estimate_log_likelihood(agents, eval_env, ckpt_filename, demo_filename)
    kl = estimate_kl(agents, eval_env, ckpt_filename, demo_filename)
    print("finishing", env_name, method)
    return [ep_len, ll, kl]


@ray.remote
def tc_run(env_name, method):
    try:
        return run(env_name, method)
    except Exception as e:
        print("exception on:", env_name, method)
        raise e


def main():
    ray.init(num_cpus=8)

    configs = []
    remotes = []
    for env_name in envs.keys():
        for method in ("RL", "BC+DC", "LFD", "MA-GAIL", "MixTURE"):
            configs.append((env_name, method))
            remotes.append(tc_run.remote(env_name, method))

    results = ray.get(remotes)
    with open("full_evaluation_results.pickle", "wb") as f:
        pickle.dump(results, f)

    # configs = [
    #     ("predatorprey_5x5", "RL"),
    #     ("predatorcapture_5x5", "RL"),
    #     ("firecommander_5x5", "RL"),
    # ]
    # results = [(1, 2, 3), (2, 3, 4), (4, 5, 6)]

    for size in (5, 10, 20):
        for method in ("RL", "BC+DC", "LFD", "MA-GAIL", "MixTURE"):
            line = []
            for task in "predatorprey", "predatorcapture", "firecommander":
                for i in range(len(configs)):
                    if configs[i] == (f"{task}_{size}x{size}", method):
                        ep_len, ll, kl = results[i]
                        line.append(f"{ep_len[0]:.2f} & {kl[0]:.3f} & {ll[0]:.2f}")
                        break
                else:
                    raise ValueError
            print(f"{size}x{size} {method}:", " & ".join(line) + r" \\")


if __name__ == "__main__":
    main()
