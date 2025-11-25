import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        input_dim = 0
        for key, space in observation_space.spaces.items():
            input_dim += space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        obs_list = []
        for key in sorted(observations.keys()):
            obs_list.append(observations[key])

        # Concatenate along feature dimension
        flat_obs = torch.cat(obs_list, dim=1)

        return self.network(flat_obs)

class AttentionNetwork(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256):
        super(AttentionNetwork, self).__init__(observation_space, features_dim)

        self.vm_encoder = nn.Sequential(
            nn.Linear(2, 64),  # vm_load + vm_available_pes
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.cloudlet_encoder = nn.Sequential(
            nn.Linear(2, 64),  # waiting_cloudlets + next_cloudlet_pes
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(256, 256),  # 128 (VM) + 128 (Cloudlet)
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations['vm_loads'].shape[0]
        num_vms = observations['vm_loads'].shape[1]

        vm_features = torch.stack([
            observations['vm_loads'],
            observations['vm_available_pes'].float()
        ], dim=2)

        vm_encoded = self.vm_encoder(vm_features)  # (batch, num_vms, 128)

        vm_attended, _ = self.attention(
            vm_encoded, vm_encoded, vm_encoded
        )  # (batch, num_vms, 128)

        vm_aggregated = vm_attended.mean(dim=1)  # (batch, 128)

        cloudlet_features = torch.cat([
            observations['waiting_cloudlets'],
            observations['next_cloudlet_pes']
        ], dim=1)  # (batch, 2)

        cloudlet_encoded = self.cloudlet_encoder(cloudlet_features)  # (batch, 128)

        combined = torch.cat([vm_aggregated, cloudlet_encoded], dim=1)  # (batch, 256)

        return self.fusion(combined)  # (batch, features_dim)

class ResidualNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ResidualNetwork, self).__init__(observation_space, features_dim)

        input_dim = sum([space.shape[0] for space in observation_space.spaces.values()])

        self.input_proj = nn.Linear(input_dim, 256)

        self.res_block1 = self._make_res_block(256, 256)
        self.res_block2 = self._make_res_block(256, 256)
        self.res_block3 = self._make_res_block(256, 256)

        self.output = nn.Linear(256, features_dim)

    def _make_res_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

    def forward(self, observations):
        obs_list = [observations[key] for key in sorted(observations.keys())]
        flat_obs = torch.cat(obs_list, dim=1)

        x = self.input_proj(flat_obs)

        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        x = x + self.res_block3(x)

        return self.output(x)