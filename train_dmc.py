import os
import argparse
from datetime import datetime
import torch

from simple_rl.env import make_dmc
from simple_rl.algorithm import PIXEL_ALGORITHMS
from simple_rl.trainer import Trainer


def run(args):
    env = make_dmc(args.domain_name, args.task_name, args.action_repeat)
    env_test = make_dmc(args.domain_name, args.task_name, args.action_repeat)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', f'{args.domain_name}-{args.task_name}',
        f'{args.algo}-seed{args.seed}-{time}')

    algo = PIXEL_ALGORITHMS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=args.seed
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        device=device,
        log_dir=log_dir,
        action_repeat=args.action_repeat,
        num_steps=args.num_steps,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=3*10**6)
    p.add_argument('--domain_name', type=str, default='cheetah')
    p.add_argument('--task_name', type=str, default='run')
    p.add_argument('--action_repeat', type=int, default=4)
    p.add_argument('--algo', type=str, default='sac_ae')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
