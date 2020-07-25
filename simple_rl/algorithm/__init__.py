from .ppo import PPO
from .ddpg import DDPG
from .td3 import TD3
from .sac import SAC
from .discor import DisCor

ALGORITHMS = {
    'ppo': PPO,
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
    'discor': DisCor
}
