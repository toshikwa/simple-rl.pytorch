from .ppo import PPO
from .ddpg import DDPG
from .sac import SAC

ALGORITHMS = {
    'ppo': PPO,
    'ddpg': DDPG,
    'sac': SAC
}
