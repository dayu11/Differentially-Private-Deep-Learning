# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from enum import Enum, auto
from typing import Iterable, Tuple

# Opting out of loading all sibling packages and their dependencies.
sys.skip_tf_privacy_import = True
from tensorflow_privacy.privacy.analysis import rdp_accountant, gdp_accountant  # noqa: E402


class RDP:
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 delta: float, orders: Iterable[float] = None) -> None:
        """
        Create a Renyi Differential Privacy accountant

        :param float noise_multiplier:
        :param float sampling_probability:
        :param float delta:
        :param Iterable[float] orders:
        """
        self.noise_multiplier = noise_multiplier
        self.sampling_probability = sampling_probability
        self.delta = delta

        if not orders:
            orders = [1.0 + x / 10.0 for x in range(1, 100)] + \
                list(range(12, 64))
        self.orders = orders

        self.rdp = rdp_accountant.compute_rdp(
            q=self.sampling_probability,
            noise_multiplier=self.noise_multiplier, steps=1,
            orders=self.orders)

    def compute_epsilon(self, num_compositions: int) -> Tuple[float, float, float]:
        rdp_steps = self.rdp*num_compositions
        eps, _, opt_order = rdp_accountant.get_privacy_spent(
            orders=self.orders, rdp=rdp_steps, target_eps=None,
            target_delta=self.delta)
        return 0.0, eps, eps


class Distribution(Enum):
    POISSON = auto()
    UNIFORM = auto()


class GDP:
    POISSON = Distribution.POISSON
    UNIFORM = Distribution.UNIFORM

    def __init__(self, noise_multiplier: float,
                 sampling_probability: float, delta: float,
                 distribution: Distribution = Distribution.UNIFORM) -> None:
        self.noise_multiplier = noise_multiplier
        self.sampling_probability = sampling_probability
        self.delta = delta

        self.distribution = distribution

    def compute_epsilon(self, num_compositions: int) -> Tuple[float, float, float]:
        batch_size = 1
        n = 1/self.sampling_probability
        epoch = num_compositions / n

        if self.distribution == Distribution.UNIFORM:
            mu = gdp_accountant.compute_mu_uniform(
                epoch, self.noise_multiplier, n, batch_size)
        elif self.distribution == Distribution.POISSON:
            mu = gdp_accountant.compute_mu_poisson(
                epoch, self.noise_multiplier, n, batch_size)
        else:
            raise ValueError()

        return 0.0, gdp_accountant.eps_from_mu(mu, self.delta), float('inf')
