import torch
from torch import nn
from torch.optim import Adam

from .sac import SAC
from simple_rl.network import (
    Encoder, Decoder, GaussianPolicyWithDetachedEncoder,
    TwinnedQFuncWithEncoder
)
from simple_rl.utils import soft_update, preprocess_states


class SACAE(SAC):

    def __init__(self, state_shape, action_shape, device, batch_size=128,
                 gamma=0.99, nstep=1, replay_size=10**6, start_steps=1000,
                 lr_encoder=1e-3, lr_decoder=1e-3, lr_actor=1e-3,
                 lr_critic=1e-3, lr_alpha=1e-4, alpha_init=0.1,
                 update_freq_actor=2, update_freq_ae=1, update_freq_target=2,
                 target_update_coef=0.01, target_update_coef_ae=0.05,
                 lambda_rae_latents=1e-6, lambda_rae_weights=1e-7):
        super().__init__(
            state_shape, action_shape, device, batch_size, gamma, nstep,
            replay_size, start_steps, lr_actor, lr_critic, lr_alpha,
            alpha_init, target_update_coef)
        assert len(state_shape) == 3

        self.optim_encoder = Adam(self.encoder.parameters(), lr=lr_encoder)
        self.optim_decoder = Adam(
            self.decoder.parameters(), lr=lr_decoder,
            weight_decay=lambda_rae_weights)

        self.update_freq_actor = update_freq_actor
        self.update_freq_ae = update_freq_ae
        self.update_freq_target = update_freq_target
        self.target_update_coef_ae = target_update_coef_ae
        self.lambda_rae_latents = lambda_rae_latents

    def build_network(self):
        self.encoder = Encoder(
            state_shape=self.state_shape,
            feature_dim=50,
            num_layers=4,
            num_filters=32
        ).to(self.device)
        self.encoder_target = Encoder(
            state_shape=self.state_shape,
            feature_dim=50,
            num_layers=4,
            num_filters=32
        ).to(self.device)
        self.decoder = Decoder(
            state_shape=self.state_shape,
            feature_dim=50,
            num_layers=4,
            num_filters=32
        ).to(self.device)
        self.actor = GaussianPolicyWithDetachedEncoder(
            encoder=self.encoder,
            action_shape=self.action_shape,
            hidden_units=[1024, 1024],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic = TwinnedQFuncWithEncoder(
            encoder=self.encoder,
            action_shape=self.action_shape,
            hidden_units=[1024, 1024],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic_target = TwinnedQFuncWithEncoder(
            encoder=self.encoder_target,
            action_shape=self.action_shape,
            hidden_units=[1024, 1024],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device).eval()

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        if self.learning_steps % self.update_freq_actor == 0:
            self.update_actor(states)
        if self.learning_steps % self.update_freq_ae == 0:
            self.update_ae(states)
        if self.learning_steps % self.update_freq_target == 0:
            self.update_target()

    def update_actor(self, states):
        with torch.no_grad():
            conv_features = self.encoder.body(states)

        actions, log_pis = self.actor.sample_without_body(conv_features)
        qs1, qs2 = self.critic.without_body(conv_features, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        loss_alpha = -self.log_alpha * (
            self.target_entropy + log_pis.detach().mean()
        )

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        self.alpha = self.log_alpha.detach().exp().item()

    def update_ae(self, states):
        # Preprocess states to be in [-0.5, 0.5] range.
        targets = preprocess_states(states)

        # Reconstruct states using autoencoder.
        features = self.encoder(states)
        reconsts = self.decoder(features)

        # MSE for reconstruction errors.
        loss_reconst = (targets - reconsts).pow_(2).mean()
        # L2 penalty of latent representations following RAE.
        loss_latent = 0.5 * features.pow(2).sum(dim=1).mean()

        # RAE loss is reconstruction loss plus the reglarizations.
        # (i.e. L2 penalty of latent representations + weight decay.)
        loss_rae = loss_reconst + self.lambda_rae_latents * loss_latent

        self.optim_encoder.zero_grad()
        self.optim_decoder.zero_grad()
        loss_rae.backward(retain_graph=False)
        self.optim_encoder.step()
        self.optim_decoder.step()

    def update_target(self):
        soft_update(
            self.critic_target.encoder, self.critic.encoder,
            self.target_update_coef_ae)
        soft_update(
            self.critic_target.mlp_critic, self.critic.mlp_critic,
            self.target_update_coef)
