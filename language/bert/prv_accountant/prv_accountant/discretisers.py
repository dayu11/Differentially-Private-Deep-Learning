# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from abc import ABC, abstractmethod

from .privacy_random_variables import PrivacyRandomVariable
from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from .domain import Domain


class Discretiser(ABC):
    @abstractmethod
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        pass


class LeftNode(Discretiser):
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        tC = domain.ts()
        tL = tC
        tR = tC + domain.dt()
        f = prv.probability(tL, tR)

        mean_d = (tC*f).sum()
        mean_c = prv.mean()

        mean_shift = mean_c - mean_d

        assert np.abs(mean_shift) < domain.dt()

        domain_shifted = domain.shift_right(mean_shift)

        return DiscretePrivacyRandomVariable(f, domain_shifted)


class CellCentred(Discretiser):
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        tC = domain.ts()
        tL = tC - domain.dt()/2.0
        tR = tC + domain.dt()/2.0
        f = prv.probability(tL, tR)

        mean_d = np.dot(tC, f)
        mean_c = prv.mean()

        mean_shift = mean_c - mean_d

        if not (np.abs(mean_shift) < domain.dt()/2):
            raise RuntimeError("Discrete mean differs from continous mean by too much.")

        domain_shifted = domain.shift_right(mean_shift)

        return DiscretePrivacyRandomVariable(f, domain_shifted)
