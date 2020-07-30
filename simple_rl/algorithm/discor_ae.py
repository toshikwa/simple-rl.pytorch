import torch
from torch import nn
from torch.optim import Adam

from .sac_ae import SACAE
from .discor import DisCor
from simple_rl.network import TwinnedErrorFuncWithEncoder
from simple_rl.utils import soft_update, disable_gradient


class DisCorAE(SACAE, DisCor):

    def __init__(self, state_shape, action_shape, device, seed, batch_size=128,
                 gamma=0.99, nstep=1, replay_size=10**6, start_steps=1000,
                 lr_encoder=1e-3, lr_decoder=1e-3, lr_actor=1e-3,
                 lr_critic=1e-3, lr_alpha=1e-4, alpha_init=0.1,
                 update_freq_actor=2, update_freq_ae=1, update_freq_target=2,
                 target_update_coef=0.01, target_update_coef_ae=0.05,
                 lambda_rae_latents=1e-6, lambda_rae_weights=1e-7,
                 lr_error=1e-3, tau_init=10.0, start_steps_is=10**4,
                 update_freq_error=2):
        super().__init__(
            state_shape, action_shape, device, seed, batch_size, gamma, nstep,
            replay_size, start_steps, lr_encoder, lr_decoder, lr_actor,
            lr_critic, lr_alpha, alpha_init, update_freq_actor, update_freq_ae,
            update_freq_target, target_update_coef, target_update_coef_ae,
            lambda_rae_latents, lambda_rae_weights)

        self.error = TwinnedErrorFuncWithEncoder(
            encoder=self.encoder,
            action_shape=self.action_shape,
            hidden_units=[1024, 1024],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.error_target = TwinnedErrorFuncWithEncoder(
            encoder=self.encoder_target,
            action_shape=self.action_shape,
            hidden_units=[1024, 1024],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device).eval()

        soft_update(self.error_target.mlp_error, self.error.mlp_error, 1.0)
        disable_gradient(self.error_target.mlp_error)

        self.optim_error = Adam(self.error.parameters(), lr=lr_error)
        self.tau1 = torch.tensor(tau_init, device=device, requires_grad=False)
        self.tau2 = torch.tensor(tau_init, device=device, requires_grad=False)

        self.start_steps_is = start_steps_is
        self.update_freq_error = update_freq_error

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        td_errors1, td_errors2 = self.update_critic_is(
            states, actions, rewards, dones, next_states
        )
        if self.learning_steps % self.update_freq_error == 0:
            self.update_error(
                states, actions, dones, next_states, td_errors1, td_errors2
            )
        if self.learning_steps % self.update_freq_actor == 0:
            self.update_actor(states)
        if self.learning_steps % self.update_freq_ae == 0:
            self.update_ae(states)
        if self.learning_steps % self.update_freq_target == 0:
            self.update_target()

    def update_target(self):
        super().update_target()
        soft_update(
            self.error_target.mlp_error,
            self.error.mlp_error,
            self.target_update_coef
        )
