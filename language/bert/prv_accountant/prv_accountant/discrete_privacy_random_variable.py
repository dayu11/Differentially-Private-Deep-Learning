# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from dataclasses import dataclass

from .domain import Domain

from typing import Tuple


@dataclass
class DiscretePrivacyRandomVariable:
    pmf: np.ndarray
    domain: Domain

    def __len__(self) -> int:
        assert len(self.pmf) == len(self.domain)
        return len(self.pmf)

    def compute_epsilon(self, delta: float, delta_error: float, epsilon_error: float) -> Tuple[float, float, float]:
        t = self.domain.ts()
        p = self.pmf
        d1 = np.flip(np.flip(p).cumsum())
        d2 = np.flip(np.flip(p*np.exp(-t)).cumsum())
        ndelta = np.exp(t) * d2-d1

        def find_epsilon(delta_target):
            i = np.searchsorted(ndelta, -delta_target, side='left')
            if i <= 0:
                raise RuntimeError("Cannot compute epsilon")
            return np.log((d1[i-1]-delta_target)/d2[i-1])
       
        eps_upper = find_epsilon(delta - delta_error) + epsilon_error
        eps_lower = find_epsilon(delta + delta_error) - epsilon_error
        eps_estimate = find_epsilon(delta)
        return float(eps_lower), float(eps_estimate), float(eps_upper)
