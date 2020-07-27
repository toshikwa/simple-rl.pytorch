# Simple RL in PyTorch
This is a simple implementation of model-free deep reinforcement learning algorithms in PyTorch. 

Currently, following algorithms have been implemented.

- State inputs
    - Proximal Policy Optimization(PPO)[[1]](#references)
    - Deep Deterministic Policy Gradient(DDPG)[[2]](#references)
    - Twin Delayed DDPG(TD3)[[3]](#references)
    - Soft Actor-Critic(SAC)[[4,5]](#references)
    - Distribution Correction(DisCor)[[6]](#references) based on SAC
- Pixel inputs
    - SAC+AE[[7]](#references)
    - DisCor+AE(SAC+AE with DisCor)

## Setup
If you are using Anaconda, first create the virtual environment.

```bash
conda create -n simple_rl python=3.8 -y
conda activate simple_rl
```

Then, you need to setup a MuJoCo license for your computer. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.

Finally, you can install Python liblaries using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## References
[[1]](https://arxiv.org/abs/1707.06347) Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[[2]](https://arxiv.org/abs/1509.02971) Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

[[3]](https://arxiv.org/abs/1802.09477) Fujimoto, Scott, Herke Van Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." arXiv preprint arXiv:1802.09477 (2018).

[[4]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[5]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

[[6]](https://arxiv.org/abs/2003.07305) Kumar, Aviral, Abhishek Gupta, and Sergey Levine. "Discor: Corrective feedback in reinforcement learning via distribution correction." arXiv preprint arXiv:2003.07305 (2020).

[[7]](https://arxiv.org/abs/1910.01741) Yarats, Denis, et al. "Improving sample efficiency in model-free reinforcement learning from images." arXiv preprint arXiv:1910.01741 (2019).
