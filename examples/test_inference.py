import dataclasses

import jax

import time
from pprint import pprint

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi05_droid")
# config = _config.get_config("pi0_aloha_sim")

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_sim")
key = jax.random.key(0)

# Create a model from the checkpoint.
model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

# We can create fake observations and actions to test the model.
obs, act = config.model.fake_obs(), config.model.fake_act()
# Sample actions from the model.
loss = model.compute_loss(key, obs, act)
print("Loss shape:", loss.shape)