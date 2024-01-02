# Mixed-Initiative Multiagent Apprenticeship Learning for Human Training of Robot Teams

## Paper Information

### Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) 2023
### Authors: Esmaeil Seraj, Jerry Xiong, Mariah Schrum, Matthew Gombolay
### Full-Read Link: https://openreview.net/forum?id=VCOZaczCHg
### Short Presentation: https://youtu.be/COGGl3lFH94?si=Z3CugC5PDTSST8gA

## Experiment: heuristic demonstrations

1. Decompress demonstration files: `cd demos ; python decompress.py`
2. Change to baselines directory: `cd ../baselines`
3. Run easy (5x5) experiments with all 4 baselines:
`PYTHONPATH=.. python run_easy_baselines.py`
4. Repeat with medium and hard

## Experiment: human study

1. Augment demonstration files: `cd demos; python augment_demos.py`
2. Change to user_study_lfd directory: `cd ../user_study_lfd`
3. `PYTHONPATH=.. python run.py`

## File Structure

The most relevant scripts are:
1. `marl.py`: defines PPO training loop, instantiated using an environment, an AgentGroup, and a RewardSignal
2. `agents/*.py`: define agent architectures, including recurrent policies, fully-connected communication, attention communication, MIM.
   - choose MIM / no MIM, attention / no attention, etc. by selecting appropriate class to instantiate
3. `reward_signals.py`: defines discriminator architectures
4. `envs/*.py`: defines environment observation/action spaces, dynamics
   - `envs/comm_wrapper.py`: defines a wrapper which adds discrete (one-hot) communication observations and actions into an existing environments
5. `ablations/simultaneous.py`: defines a combined PPO+BC trainer which adds BC term to the loss during online updates
6. `expert_heuristics/*.py`: used to create heuristic demonstration datasets
7. `ez_tuning.py`: defines hyperparameter tuning framework + statistics (IQM, boostrapped confidence intervals)

## Major tunable hyperparameters:

- `lr`: learning rate for policy and critic can be specified when instantiating an AgentGroup
  - learning rate for the discriminator can be specified when instantiating the appropriate RewardSignal
  - note that CombinedTrainer constructor overrides AgentGroup-specified learning rates.
- `fc_dim`: the hidden dimensionality (width) of policy and critic architectures, can be
specified when instantiated an AgentGroup

## Examples

### Running our method w/o behavioral cloning

```python
from agents import FullAttentionMIMAgents
from envs import FireCommander5x5
from reward_signals import MixedGAILReward
from marl import PPOTrainer

agents = FullAttentionMIMAgents(1e-3, 1e-3, mim_coeff=0.01, fc_dim=64)
reward = MixedGAILReward("demos/firecommander_5x5.pickle", lr=1e-5)

trainer = PPOTrainer(
    FireCommander5x5(),
    agents,
    reward.normalized(),
    gae_lambda=0.5,
    minibatch_size=32,
)

for _ in range(100):
    trainer.run()

trainer.evaluate()
print(trainer.logger.data["episode_len"][-1])
```

### Running our method w/ behavioral cloning

```python
from ablations.simultaneous import CombinedTrainer, ExposedBCTrainer, ExposedPPOTrainer
from agents import FullAttentionMIMAgents
from envs import FireCommander10x10
from reward_signals import MixedGAILReward

demo_filename = "demos/firecommander_10x10.pickle"
agents = FullAttentionMIMAgents(0, 0, mim_coeff=0.01, fc_dim=64)
env = FireCommander10x10(n_fires=1)
reward = MixedGAILReward(demo_filename, lr=1e-5)

trainer = CombinedTrainer(
    lr = 1e-3,
    bc_trainer=ExposedBCTrainer(
        env,
        agents,
        demo_filename=demo_filename,
        minibatch_size=32,
    ),
    ppo_trainer=ExposedPPOTrainer(
        env,
        agents,
        reward.normalized(),
        gae_lambda=0.5,
        minibatch_size=32,
    ),
    bc_weight=0.1,
)

for _ in range(100):
    trainer.run()

trainer.evaluate()
print(trainer.logger.data["episode_len"][-1])
```
### Questions

In case of any questions, please reach out directly to Esmaeil Seraj at <eseraj3@gatech.edu>

### Citation

```
@inproceedings{seraj2023mixed,
  title={Mixed-Initiative Multiagent Apprenticeship Learning for Human Training of Robot Teams},
  author={Seraj, Esmaeil and Xiong, Jerry Yuyang and Schrum, Mariah L and Gombolay, Matthew},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
