from .ppo import PPO
from .ddpg import DDPG
from .td3 import TD3
from .sac import SAC
from .discor import DisCor
from .sac_ae import SACAE

STATE_ALGORITHMS = {
    'ppo': PPO,
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
    'discor': DisCor
}

PIXEL_ALGORITHMS = {
    'sac_ae': SACAE
}
