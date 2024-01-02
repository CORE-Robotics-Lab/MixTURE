from .base_multi_agent import MultiAgentEnv, ObsType
from .comm_wrapper import CommWrapper, add_uniform_comms
from .fire_commander import FireCommander
from .predator_capture import PredatorCapture
from .predator_prey import PredatorPrey
from .simple_fire_commander import SimpleFireCommander


class PredatorPrey5x5(PredatorPrey):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=5,
            n_predator=3,
            n_prey=1,
            vision=0,
            max_timesteps=20,
            **kwargs,
        )


class PredatorCapture5x5(PredatorCapture):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=5,
            n_predator=2,
            n_capture=1,
            n_prey=1,
            vision=0,
            max_timesteps=40,
            **kwargs,
        )


class FireCommander5x5(FireCommander):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=5,
            n_perception=2,
            n_action=1,
            n_fires=1,
            vision=1,
            max_timesteps=80,
            **kwargs,
        )


class PredatorPrey10x10(PredatorPrey):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=10,
            n_predator=6,
            n_prey=1,
            vision=1,
            max_timesteps=80,
            **kwargs,
        )


class PredatorCapture10x10(PredatorCapture):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=10,
            n_predator=3,
            n_capture=3,
            n_prey=1,
            vision=1,
            max_timesteps=80,
            **kwargs,
        )


class FireCommander10x10(FireCommander):
    def __init__(self, n_fires, **kwargs):
        super().__init__(
            map_size=10,
            n_perception=3,
            n_action=3,
            n_fires=n_fires,
            vision=1,
            max_timesteps=80,
            **kwargs,
        )


class PredatorPrey20x20(PredatorPrey):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=20,
            n_predator=10,
            n_prey=1,
            vision=2,
            max_timesteps=80,
            **kwargs,
        )


class PredatorCapture20x20(PredatorCapture):
    def __init__(self, **kwargs):
        super().__init__(
            map_size=20,
            n_predator=6,
            n_capture=4,
            n_prey=1,
            vision=2,
            max_timesteps=80,
            **kwargs,
        )


class FireCommander20x20(FireCommander):
    def __init__(self, n_fires, **kwargs):
        super().__init__(
            map_size=20,
            n_perception=6,
            n_action=4,
            n_fires=n_fires,
            vision=2,
            max_timesteps=80,
            **kwargs,
        )
