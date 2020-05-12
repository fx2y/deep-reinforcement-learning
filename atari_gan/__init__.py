import random

import numpy as np
import torch

from .input_wrapper import InputWrapper
from .discriminator import Discriminator
from .generator import Generator, LATENT_VECTOR_SIZE

BATCH_SIZE = 16


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)
    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32)
            batch_np *= 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


__all__ = [
    'InputWrapper',
    'Discriminator',
    'Generator',
    'iterate_batches',
    'BATCH_SIZE',
    'LATENT_VECTOR_SIZE'
]
